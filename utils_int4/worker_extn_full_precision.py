import torch
import vllm
from collections import deque
from utils_int4.comm import _stateless_init_process_group

# QuZO helpers

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

def _get_stochastic_perturbation(shape, device, noise_seed, noise_sigma, round_seed, bits):
    """Generate a quantized stochastic perturbation (delta) tensor.

    IMPORTANT: In this repo, the packed `qweight` nibbles/bytes are treated as
    *packed codes* in `[0 .. 2^bits - 1]` (see `_pack_qweight(...).clamp(0, 15)`
    for int4). For exact restore, we must avoid any lossy clamping during pack,
    so boundary gating must use the same code-range.
    """

    # Mirror sampling: negative `noise_seed` indicates we should negate the noise
    # while preserving the same RNG stream as `abs(noise_seed)`.
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

    code_max = (1 << int(bits)) - 1
    max_delta = int(code_max)
    u_clamped = torch.clamp(u_quantized, min=-max_delta, max=max_delta)
    return u_clamped.to(dtype=torch.int32)


def quzo_perturb(w_int, noise_seed, noise_sigma, round_seed, bits):
    """Perturb packed integer *codes* with boundary-only gating.

    Returns `(w_perturbed, boundary_mask)` where `boundary_mask == True` means
    "skip perturbation" at that element because it would push the packed code
    out of range.

    NOTE: `w_int` is expected to be in the packed-code domain:
      - int4: [0..15]
      - int8: [0..255]
    This matches `_unpack_qweight`/`_pack_qweight` in this file.
    """

    u_q = _get_stochastic_perturbation(
        w_int.shape, w_int.device, noise_seed, noise_sigma, round_seed, bits
    )

    limit_min = 0
    limit_max = (1 << int(bits)) - 1

    w_i32 = w_int.to(torch.int32)
    w_candidate = w_i32 + u_q
    boundary_mask = (w_candidate > limit_max) | (w_candidate < limit_min)

    u_effective = u_q.masked_fill(boundary_mask, 0)
    w_perturbed = w_i32 + u_effective
    return w_perturbed, boundary_mask


def quzo_restore(w_perturbed, boundary_mask, noise_seed, noise_sigma, round_seed, bits):
    """Restore exact packed-code values using the saved boundary mask."""

    u_q = _get_stochastic_perturbation(
        w_perturbed.shape, w_perturbed.device, noise_seed, noise_sigma, round_seed, bits
    )
    u_effective = u_q.masked_fill(boundary_mask, 0)
    return w_perturbed.to(torch.int32) - u_effective

class BaseESExtension:
    """
    Scale-aware dual-evolution for GPTQ-Marlin int4 models.
    - Perturbs packed integer codes with boundary-only gating
    - Perturbs floating-point scales multiplicatively in log-space
    - Applies discrete ES updates with optional momentum + scale updates
    """

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int = 0):
        if not hasattr(self, "inter_pg"):
            raise RuntimeError("Process group not initialized. Call init_inter_engine_group first.")

        stream = torch.cuda.current_stream()
        seen_ptrs = set()

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
        for _, module in self._iter_gptq_modules():
            for attr in ["qweight", "qzeros", "scales", "g_idx", "g_idx_sort_indices"]:
                if hasattr(module, attr):
                    _broadcast_tensor(getattr(module, attr))

        torch.cuda.synchronize()
        return True

    def _iter_gptq_modules(self):
        for name, module in self.model_runner.model.named_modules():
            if hasattr(module, "quant_method") and isinstance(
                module.quant_method, vllm.model_executor.layers.quantization.gptq_marlin.GPTQMarlinLinearMethod
            ):
                yield name, module

    def _get_w_bits(self, module) -> int:
        """Best-effort detection of weight bitwidth; defaults to 4 for int4."""
        for attr in ("w_bit", "w_bits", "bits"):
            if hasattr(module, attr):
                try:
                    val = int(getattr(module, attr))
                    if val in (4, 8):
                        return val
                except Exception:
                    pass
        if hasattr(module, "quant_method") and hasattr(module.quant_method, "w_bit"):
            try:
                val = int(module.quant_method.w_bit)
                if val in (4, 8):
                    return val
            except Exception:
                pass
        return 4

    def _unpack_qweight(self, qweight: torch.Tensor, w_bits: int) -> torch.Tensor:
        if w_bits == 8:
            unpacked = torch.stack([(qweight >> (8 * i)) & 0xFF for i in range(4)], dim=-1)
        elif w_bits == 4:
            unpacked = torch.stack([(qweight >> (4 * i)) & 0xF for i in range(8)], dim=-1)
        else:
            unpacked = torch.stack([(qweight >> (8 * i)) & 0xFF for i in range(4)], dim=-1)
        return unpacked.to(torch.int16)

    def _pack_qweight(self, unpacked: torch.Tensor, original_dtype: torch.dtype, w_bits: int) -> torch.Tensor:
        if w_bits == 8:
            w = unpacked.clamp(0, 255).to(torch.int32)
            packed = w[..., 0] | (w[..., 1] << 8) | (w[..., 2] << 16) | (w[..., 3] << 24)
        elif w_bits == 4:
            w = unpacked.clamp(0, 15).to(torch.int32)
            packed = (
                w[..., 0]
                | (w[..., 1] << 4)
                | (w[..., 2] << 8)
                | (w[..., 3] << 12)
                | (w[..., 4] << 16)
                | (w[..., 5] << 20)
                | (w[..., 6] << 24)
                | (w[..., 7] << 28)
            )
        else:
            w = unpacked.clamp(0, 255).to(torch.int32)
            packed = w[..., 0] | (w[..., 1] << 8) | (w[..., 2] << 16) | (w[..., 3] << 24)
        return packed.to(original_dtype)

    def _gen_noise(self, seed: int, shape: tuple, device: torch.device, p_zero: float) -> torch.Tensor:
        base_seed, sign = _mirror_seed(seed)
        gen = torch.Generator(device=device).manual_seed(int(base_seed))
        probs = torch.rand(shape, generator=gen, device=device)
        noise = torch.zeros(shape, dtype=torch.int16, device=device)
        threshold = (1.0 - p_zero) / 2.0
        noise[probs < threshold] = -1
        noise[probs > (1.0 - threshold)] = 1
        if sign < 0:
            noise = -noise
        return noise

    @torch.no_grad()
    def perturb_self_weights(self, seed: int, sigma: float = 1.0):
        if not hasattr(self, "_noise_state"):
            self._noise_state = {}
        state_key = (seed, sigma)
        self._noise_state[state_key] = {}

        total_changes = 0
        total_params = 0

        for name, module in self._iter_gptq_modules():
            # print(f"Perturbing module: {name}")
            dev = module.qweight.device
            w_bits = self._get_w_bits(module)

            # Integer perturbation with boundary-only gating
            w_int = self._unpack_qweight(module.qweight, w_bits)
            # Mirror sampling: if seed is negative, use the same RNG stream as abs(seed)
            # but negate the sampled perturbations. To keep the stochastic rounding stream
            # mirrored as well, we also mirror the derived round seed.
            round_seed = seed + 12345 if int(seed) >= 0 else -(abs(int(seed)) + 12345)
            w_new, boundary_mask = quzo_perturb(
                w_int,
                noise_seed=seed,
                noise_sigma=sigma,
                round_seed=round_seed,
                bits=w_bits
            )

            self._noise_state[state_key][name] = boundary_mask

            module.qweight.copy_(self._pack_qweight(w_new, module.qweight.dtype, w_bits))

            n_diff = (w_new != w_int).sum().item()
            total_changes += n_diff
            total_params += w_new.numel()

        torch.cuda.empty_cache()
        ratio = total_changes / total_params if total_params > 0 else 0
        return {"weight_change_ratio": ratio, "total_changes": total_changes, "total_params": total_params}

    @torch.no_grad()
    def restore_self_weights(self, seed: int, sigma: float = 1.0):
        # 1. Retrieve the state using the exact same key
        state_key = (seed, sigma)
        if state_key not in getattr(self, "_noise_state", {}):
            raise RuntimeError(f"No saved state found for seed {seed}")

        saved_masks = self._noise_state.pop(state_key)

        for name, module in self._iter_gptq_modules():
            dev = module.qweight.device
            w_bits = self._get_w_bits(module)

            # 2. Retrieve the boundary mask saved during perturbation
            # Recall: boundary_mask is True where noise was SKIPPED (clipped)
            boundary_mask = saved_masks[name].to(dev)

            # 3. Unpack current (perturbed) weights
            w_curr = self._unpack_qweight(module.qweight, w_bits)

            # 4. Restore using quzo_restore
            # CRITICAL: round_seed must match perturb_self_weights, including mirroring.
            round_seed = seed + 12345 if int(seed) >= 0 else -(abs(int(seed)) + 12345)
            w_restored = quzo_restore(
                w_perturbed=w_curr,
                boundary_mask=boundary_mask,
                noise_seed=seed,
                noise_sigma=sigma,
                round_seed=round_seed,
                bits=w_bits
            )

            # 5. Pack and copy back
            module.qweight.copy_(self._pack_qweight(w_restored, module.qweight.dtype, w_bits))

        torch.cuda.empty_cache()
        return True

    # WORKING VERSION
    @torch.no_grad()
    def apply_quzo_perturb_update(
        self,
        per_seed_coeffs: list,
        sigma: float = 0.1,
        alpha: float = 0.001,
        keep_residual: bool = True,
        residual_decay: float = 0.9,
    ):
        if keep_residual and not hasattr(self, "_residuals"):
            self._residuals = {}

        total_update_magnitude = 0.0
        weight_change = 0
        boundary_hits = 0
        attempted_updates = 0
        total_params = 0

        # Pre-calculate scale factor to normalize noise magnitude
        # grad ~ sum(coeff * noise) / sigma
        scale_factor = 1.0 / max(1e-8, float(sigma))

        for name, module in self._iter_gptq_modules():
            dev = module.qweight.device
            w_bits = self._get_w_bits(module)
            limit_max = (1 << int(w_bits)) - 1

            # 1. Unpack current weights
            w_int = self._unpack_qweight(module.qweight, w_bits).to(torch.int32)
            
            # 2. Gradient Estimation
            grad_est = torch.zeros_like(w_int, dtype=torch.float32)
            
            for seed, coeff in per_seed_coeffs:
                seed_i = int(seed)
                # Ensure correct mirroring logic matches perturbation
                round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)

                u_q = _get_stochastic_perturbation(
                    w_int.shape, dev, 
                    noise_seed=seed_i, noise_sigma=sigma, round_seed=round_seed, bits=w_bits
                ).to(torch.int32)

                # Re-apply boundary gating
                w_candidate = w_int + u_q
                boundary_mask = (w_candidate > limit_max) | (w_candidate < 0)
                u_eff = u_q.masked_fill(boundary_mask, 0)

                grad_est.add_(u_eff.to(torch.float32), alpha=float(coeff))

            grad_est.mul_(scale_factor)
            
            # Calculate the "Desired Float Step"
            float_step = grad_est.mul(alpha)

            # 3. Add Residuals with PHASE SHIFT
            if keep_residual:
                # PHASE SHIFT INITIALIZATION
                # If this is the first time touching this layer, init residuals randomly.
                # This prevents "Update Synchronization" where all weights flip at step N.
                if name not in self._residuals:
                    # Random noise in [-0.5, 0.5] ensures params are at different
                    # stages of their accumulation cycle.
                    self._residuals[name] = (torch.rand_like(float_step) - 0.5).to(torch.float16)

                prev_resid = self._residuals[name].to(dev).float()
                
                # Leaky Integration
                float_step.add_(prev_resid, alpha=residual_decay)

            # 4. Discrete Update
            # We use standard rounding. Because of Phase Shift, this is now safe.
            delta_int = torch.round(float_step).to(torch.int32)

            attempted_updates += (delta_int != 0).sum().item()

            # 5. Anti-Windup & Boundary Check
            w_new = w_int + delta_int
            valid_mask = (w_new >= 0) & (w_new <= limit_max)

            # Count how many elements would cross the code boundary (i.e., are blocked).
            # We only count attempted moves (delta != 0).
            boundary_hits += ((~valid_mask) & (delta_int != 0)).sum().item()

            delta_final = torch.where(valid_mask, delta_int, torch.tensor(0, device=dev, dtype=torch.int32))
            w_final = w_int + delta_final

            # 6. Update Residuals
            if keep_residual:
                # Residual = (Accumulated Float) - (Applied Integer)
                new_residual = float_step - delta_final.to(torch.float32)
                
                new_residual.clamp_(-1.5, 1.5)
                
                self._residuals[name] = new_residual.cpu().to(torch.float16)

            # 7. Pack and Write
            module.qweight.copy_(self._pack_qweight(w_final, module.qweight.dtype, w_bits))

            weight_change += (delta_final != 0).sum().item()
            total_params += w_int.numel()
            total_update_magnitude += float_step.abs().mean().item()

        torch.cuda.empty_cache()
        ratio = weight_change / total_params if total_params > 0 else 0.0

        boundary_ratio = (
            boundary_hits / attempted_updates if attempted_updates > 0 else 0.0
        )
        
        # Debug info is crucial for tuning decay/alpha
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