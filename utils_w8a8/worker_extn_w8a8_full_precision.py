import torch
from utils_int4.comm import _stateless_init_process_group


def _mirror_seed(seed: int) -> tuple[int, int]:
    """Return (base_seed, sign) for mirror sampling.

    Convention:
    - If `seed >= 0`: base_seed=seed, sign=+1
    - If `seed < 0`:  base_seed=abs(seed), sign=-1  (mirror sample)

    Using base_seed keeps RNG streams identical between +s and -s.
    """
    seed_i = int(seed)
    if seed_i < 0:
        return -seed_i, -1
    return seed_i, 1


def _get_stochastic_perturbation_int8_signed(
    shape,
    device,
    noise_seed: int,
    noise_sigma: float,
    round_seed: int,
    *,
    max_abs_delta: int = 127,
) -> torch.Tensor:
    """Generate a quantized stochastic perturbation (delta) tensor for signed int8 code space.

    This mirrors the int4 implementation:
    - Sample Gaussian noise
    - Scale by sigma
    - Stochastic round to integer (floor + Bernoulli(fraction))
    - Mirror sampling: negative seed negates noise but keeps RNG stream

    Returns int32 tensor.
    """

    base_noise_seed, sign = _mirror_seed(int(noise_seed))
    base_round_seed, _ = _mirror_seed(int(round_seed))

    gen_noise = torch.Generator(device=device)
    gen_noise.manual_seed(int(base_noise_seed))

    gen_round = torch.Generator(device=device)
    gen_round.manual_seed(int(base_round_seed))

    u_float = torch.randn(shape, generator=gen_noise, device=device)
    scaled_u = u_float * float(noise_sigma)

    floor_u = torch.floor(scaled_u)
    fractional_part = scaled_u - floor_u  # in [0, 1)
    bernoulli_sample = torch.bernoulli(fractional_part, generator=gen_round)
    u_quantized = floor_u + bernoulli_sample

    if sign < 0:
        u_quantized = -u_quantized

    u_clamped = torch.clamp(
        u_quantized,
        min=-int(max_abs_delta),
        max=int(max_abs_delta),
    )
    return u_clamped.to(dtype=torch.int32)


def quzo_perturb_int8_signed(
    w_int8: torch.Tensor,
    *,
    noise_seed: int,
    noise_sigma: float,
    round_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb signed int8 weights with boundary-only gating.

    Returns `(w_perturbed_int8, boundary_mask)` where `boundary_mask == True` means
    "skip perturbation" at that element because it would push the code out of range.

    Signed int8 domain is [-128, 127].
    """

    if w_int8.dtype != torch.int8:
        raise TypeError(f"Expected int8 weights, got {w_int8.dtype}")

    u_q = _get_stochastic_perturbation_int8_signed(
        w_int8.shape,
        w_int8.device,
        noise_seed=int(noise_seed),
        noise_sigma=float(noise_sigma),
        round_seed=int(round_seed),
    )

    limit_min = -128
    limit_max = 127

    w_i32 = w_int8.to(torch.int32)
    w_candidate = w_i32 + u_q
    boundary_mask = (w_candidate > limit_max) | (w_candidate < limit_min)

    u_effective = u_q.masked_fill(boundary_mask, 0)
    w_perturbed = w_i32 + u_effective

    return w_perturbed.to(torch.int8), boundary_mask


def quzo_restore_int8_signed(
    w_perturbed_int8: torch.Tensor,
    boundary_mask: torch.Tensor,
    *,
    noise_seed: int,
    noise_sigma: float,
    round_seed: int,
) -> torch.Tensor:
    """Restore exact signed int8 values using the saved boundary mask."""

    if w_perturbed_int8.dtype != torch.int8:
        raise TypeError(f"Expected int8 weights, got {w_perturbed_int8.dtype}")

    u_q = _get_stochastic_perturbation_int8_signed(
        w_perturbed_int8.shape,
        w_perturbed_int8.device,
        noise_seed=int(noise_seed),
        noise_sigma=float(noise_sigma),
        round_seed=int(round_seed),
    )
    u_effective = u_q.masked_fill(boundary_mask, 0)

    w_restored = w_perturbed_int8.to(torch.int32) - u_effective
    return w_restored.to(torch.int8)


class W8A8ESExtension:
    """QuZO-style perturb/restore for W8A8 models.

    This extension targets models where linear weights are stored directly as `torch.int8`
    tensors (no packing like GPTQ int4). We perturb in signed int8 code space with:
    - mirror sampling (negative seeds)
    - boundary-only gating for exact restore
    """

    def _iter_int8_weight_modules(self):
        """Iterate modules whose int8 weights we want to optimize.

        Important: unlike the int4 GPTQ-Marlin path (which iterates only quantized
        Linear layers), a naive "any module with int8 weight" iterator can catch
        unintended tensors (e.g., embeddings or other quantized buffers) and
        quickly destabilize ES.

        Here we conservatively target Linear-like modules only.
        """

        for name, module in self.model_runner.model.named_modules():
            if not hasattr(module, "weight"):
                continue
            w = getattr(module, "weight")
            if not isinstance(w, torch.Tensor) or w.dtype != torch.int8:
                continue

            # Only touch 2D "Linear"-like weights.
            # This matches the intended scope seen in vLLM W8A8 models
            # (e.g., MergedColumnParallelLinear / RowParallelLinear).
            if w.dim() != 2:
                continue
            cls_name = module.__class__.__name__
            if "Linear" not in cls_name:
                continue

            yield name, module

    def init_inter_engine_group(
        self, master_address: str, master_port: int, rank: int, world_size: int
    ):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int = 0):
        if not hasattr(self, "inter_pg"):
            raise RuntimeError(
                "Process group not initialized. Call init_inter_engine_group first."
            )

        stream = torch.cuda.current_stream()
        seen_ptrs: set[int] = set()

        def _broadcast_tensor(t: torch.Tensor):
            if t is None or not isinstance(t, torch.Tensor):
                return
            if t.data_ptr() in seen_ptrs:
                return
            seen_ptrs.add(t.data_ptr())
            if not t.is_cuda:
                t_cuda = t.to(self.device, non_blocking=True)
                self.inter_pg.broadcast(t_cuda, src=src_rank, stream=stream)
                t.copy_(t_cuda.cpu(), non_blocking=True)
            else:
                self.inter_pg.broadcast(t, src=src_rank, stream=stream)

        for _, p in self.model_runner.model.named_parameters():
            _broadcast_tensor(p)
        for _, b in self.model_runner.model.named_buffers():
            _broadcast_tensor(b)
        for _, module in self._iter_int8_weight_modules():
            _broadcast_tensor(module.weight)

        torch.cuda.synchronize()
        return True

    @torch.no_grad()
    def perturb_self_weights(self, seed: int, sigma: float = 1.0):
        if not hasattr(self, "_noise_state"):
            self._noise_state = {}

        state_key = (int(seed), float(sigma))
        self._noise_state[state_key] = {}

        total_changes = 0
        total_params = 0

        # Keep round_seed mirroring consistent with int4 code.
        seed_i = int(seed)
        round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)

        with torch.no_grad():
            for name, module in self._iter_int8_weight_modules():
                w = module.weight

                w_new, boundary_mask = quzo_perturb_int8_signed(
                    w,
                    noise_seed=seed_i,
                    noise_sigma=float(sigma),
                    round_seed=round_seed,
                )

                # Save mask on CPU to reduce GPU memory.
                self._noise_state[state_key][name] = boundary_mask.cpu()

                total_changes += (w_new != w).sum().item()
                total_params += w.numel()

                module.weight.copy_(w_new)

        torch.cuda.empty_cache()
        ratio = total_changes / total_params if total_params > 0 else 0.0
        return {
            "weight_change_ratio": ratio,
            "total_changes": int(total_changes),
            "total_params": int(total_params),
        }

    @torch.no_grad()
    def restore_self_weights(self, seed: int, sigma: float = 1.0):
        state_key = (int(seed), float(sigma))
        if state_key not in getattr(self, "_noise_state", {}):
            raise RuntimeError(f"No saved state found for seed={seed} sigma={sigma}")

        saved_masks = self._noise_state.pop(state_key)

        seed_i = int(seed)
        round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)

        restored = 0
        with torch.no_grad():
            for name, module in self._iter_int8_weight_modules():
                if name not in saved_masks:
                    continue
                w = module.weight
                boundary_mask = saved_masks[name].to(w.device)

                w_restored = quzo_restore_int8_signed(
                    w,
                    boundary_mask,
                    noise_seed=seed_i,
                    noise_sigma=float(sigma),
                    round_seed=round_seed,
                )
                module.weight.copy_(w_restored)
                restored += 1

        torch.cuda.empty_cache()
        return {"restored_modules": int(restored)}

    @torch.no_grad()
    def apply_quzo_perturb_update(
        self,
        per_seed_coeffs: list,
        sigma: float = 0.1,
        alpha: float = 0.001,
        keep_residual: bool = True,
        residual_decay: float = 0.9,
    ):
        """Apply a QuZO-style discrete ES update to signed int8 weights.

        Mirrors `utils_int4/worker_extn_full_precision.py::apply_quzo_perturb_update`,
        but operates directly in signed int8 code space [-128, 127].
        """

        if keep_residual and not hasattr(self, "_residuals"):
            self._residuals = {}

        total_update_magnitude = 0.0
        weight_change = 0
        boundary_hits = 0
        attempted_updates = 0
        total_params = 0

        scale_factor = 1.0 / max(1e-8, float(sigma))
        limit_min = -128
        limit_max = 127

        with torch.no_grad():
            for name, module in self._iter_int8_weight_modules():
                w = module.weight
                dev = w.device

                w_int = w.to(torch.int32)
                grad_est = torch.zeros_like(w_int, dtype=torch.float32)

                for seed, coeff in per_seed_coeffs:
                    seed_i = int(seed)
                    round_seed = (
                        seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)
                    )

                    u_q = _get_stochastic_perturbation_int8_signed(
                        w_int.shape,
                        dev,
                        noise_seed=seed_i,
                        noise_sigma=float(sigma),
                        round_seed=round_seed,
                        max_abs_delta=127,
                    ).to(torch.int32)

                    w_candidate = w_int + u_q
                    boundary_mask = (w_candidate > limit_max) | (w_candidate < limit_min)
                    u_eff = u_q.masked_fill(boundary_mask, 0)

                    grad_est.add_(u_eff.to(torch.float32), alpha=float(coeff))

                grad_est.mul_(scale_factor)
                float_step = grad_est.mul(alpha)

                if keep_residual:
                    if name not in self._residuals:
                        self._residuals[name] = (
                            torch.rand_like(float_step) - 0.5
                        ).to(torch.float16)

                    prev_resid = self._residuals[name].to(dev).float()
                    float_step.add_(prev_resid, alpha=residual_decay)

                delta_int = torch.round(float_step).to(torch.int32)

                attempted_updates += (delta_int != 0).sum().item()

                w_new = w_int + delta_int
                valid_mask = (w_new >= limit_min) & (w_new <= limit_max)

                # Count how many elements would cross the code boundary (i.e., are blocked).
                # We only count attempted moves (delta != 0).
                boundary_hits += ((~valid_mask) & (delta_int != 0)).sum().item()

                delta_final = torch.where(
                    valid_mask,
                    delta_int,
                    torch.tensor(0, device=dev, dtype=torch.int32),
                )
                w_final = w_int + delta_final

                if keep_residual:
                    new_residual = float_step - delta_final.to(torch.float32)
                    new_residual.clamp_(-1.5, 1.5)
                    self._residuals[name] = new_residual.cpu().to(torch.float16)

                module.weight.copy_(w_final.to(torch.int8))

                weight_change += (delta_final != 0).sum().item()
                total_params += w_int.numel()
                total_update_magnitude += float_step.abs().mean().item()

        torch.cuda.empty_cache()
        ratio = weight_change / total_params if total_params > 0 else 0.0
        boundary_ratio = (
            boundary_hits / attempted_updates if attempted_updates > 0 else 0.0
        )
        print(
            f"Update (Decay={residual_decay}). Sparsity: {ratio:.6f}, "
            f"BoundaryHit: {boundary_ratio:.6f} ({int(boundary_hits)}/{int(attempted_updates)}), "
            f"AvgStep: {total_update_magnitude:.1e}"
        )
        return {
            "weight_change_ratio": ratio,
            "boundary_hits": int(boundary_hits),
            "boundary_hit_ratio": float(boundary_ratio),
            "attempted_updates": int(attempted_updates),
        }
