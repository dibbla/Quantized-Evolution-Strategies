import torch
import vllm
from collections import deque
from utils_int4.comm import _stateless_init_process_group

# --- QuZO Specific Helpers ---

def _mirror_seed(seed: int) -> tuple[int, int]:
    """Return (base_seed, sign) for mirror sampling."""
    seed_i = int(seed)
    if seed_i < 0:
        return -seed_i, -1
    return seed_i, 1

def _stochastic_round_tensor(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply stochastic rounding to a float tensor: floor(x) + Bernoulli(x - floor(x))."""
    floor_t = torch.floor(tensor)
    fractional_part = tensor - floor_t
    bernoulli_sample = torch.bernoulli(fractional_part, generator=generator)
    return floor_t + bernoulli_sample

def _get_stochastic_perturbation(shape, device, noise_seed, noise_sigma, round_seed, bits):
    """
    Generate a quantized stochastic perturbation.
    
    QuZO Requirement: Separation of `noise_seed` (continuous noise) and `round_seed` (quantization).
    """
    # Mirror logic for noise seed
    base_noise_seed, sign_noise = _mirror_seed(int(noise_seed))
    # Mirror logic for round seed
    base_round_seed, _ = _mirror_seed(int(round_seed))

    gen_noise = torch.Generator(device=device)
    gen_noise.manual_seed(int(base_noise_seed))

    gen_round = torch.Generator(device=device)
    gen_round.manual_seed(int(base_round_seed))

    # 1. Generate Continuous Noise u_i [cite: 108]
    u_float = torch.randn(shape, generator=gen_noise, device=device)
    scaled_u = u_float * float(noise_sigma)

    # 2. Stochastic Rounding Q(u_i) [cite: 137]
    u_quantized = _stochastic_round_tensor(scaled_u, gen_round)
    
    # Apply sign flip if mirrored noise
    if sign_noise < 0:
        u_quantized = -u_quantized

    # Clamp to code range (0..15 for INT4)
    code_max = (1 << int(bits)) - 1
    max_delta = int(code_max)
    u_clamped = torch.clamp(u_quantized, min=-max_delta, max=max_delta)
    
    return u_clamped.to(dtype=torch.int32)

def quzo_perturb(w_int, noise_seed, noise_sigma, round_seed, bits):
    """
    Generates u_{i,1} for the forward pass[cite: 145].
    """
    u_q = _get_stochastic_perturbation(
        w_int.shape, w_int.device, noise_seed, noise_sigma, round_seed, bits
    )

    limit_min = 0
    limit_max = (1 << int(bits)) - 1

    w_i32 = w_int.to(torch.int32)
    w_candidate = w_i32 + u_q
    
    # Boundary gating (mask out perturbations that overflow code range)
    boundary_mask = (w_candidate > limit_max) | (w_candidate < limit_min)

    u_effective = u_q.masked_fill(boundary_mask, 0)
    w_perturbed = w_i32 + u_effective
    return w_perturbed, boundary_mask

def quzo_restore(w_perturbed, boundary_mask, noise_seed, noise_sigma, round_seed, bits):
    """Restores weights by subtracting u_{i,1}."""
    u_q = _get_stochastic_perturbation(
        w_perturbed.shape, w_perturbed.device, noise_seed, noise_sigma, round_seed, bits
    )
    u_effective = u_q.masked_fill(boundary_mask, 0)
    return w_perturbed.to(torch.int32) - u_effective


class BaselineQuZOExtension:
    """
    QuZO: Quantized Zeroth-Order Fine-Tuning Implementation.
    Implements Q-RGE2 gradient estimation and stochastic updates.
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

    @torch.no_grad()
    def perturb_self_weights(self, seed: int, sigma: float = 1.0):
        """
        Applies u_{i,1} perturbation for forward pass.
        Corresponds to Eq 7: u_{i,1} = Q1(u_i)
        """
        if not hasattr(self, "_noise_state"):
            self._noise_state = {}
        state_key = (seed, sigma)
        self._noise_state[state_key] = {}

        total_changes = 0
        total_params = 0

        # SEED STRATEGY: 
        # noise_seed = seed
        # round_seed_1 = seed + constant_1 (for perturbation)
        round_seed_1 = seed + 12345 if int(seed) >= 0 else -(abs(int(seed)) + 12345)

        for name, module in self._iter_gptq_modules():
            w_bits = self._get_w_bits(module)
            w_int = self._unpack_qweight(module.qweight, w_bits)
            
            w_new, boundary_mask = quzo_perturb(
                w_int,
                noise_seed=seed,
                noise_sigma=sigma,
                round_seed=round_seed_1,
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
        """Restores weights to original state."""
        state_key = (seed, sigma)
        if state_key not in getattr(self, "_noise_state", {}):
            raise RuntimeError(f"No saved state found for seed {seed}")

        saved_masks = self._noise_state.pop(state_key)
        round_seed_1 = seed + 12345 if int(seed) >= 0 else -(abs(int(seed)) + 12345)

        for name, module in self._iter_gptq_modules():
            w_bits = self._get_w_bits(module)
            boundary_mask = saved_masks[name].to(module.qweight.device)
            w_curr = self._unpack_qweight(module.qweight, w_bits)

            w_restored = quzo_restore(
                w_perturbed=w_curr,
                boundary_mask=boundary_mask,
                noise_seed=seed,
                noise_sigma=sigma,
                round_seed=round_seed_1,
                bits=w_bits
            )

            module.qweight.copy_(self._pack_qweight(w_restored, module.qweight.dtype, w_bits))

        torch.cuda.empty_cache()
        return True

    # --- IMPLEMENTATION OF QUZO UPDATE ---
    @torch.no_grad()
    def apply_quzo_perturb_update(
        self,
        per_seed_coeffs: list,
        sigma: float = 0.1,
        alpha: float = 0.001,
        update_seed: int = 42,
        *args, **kwargs
    ):
        """
        Strict QuZO Update Step (Eq. 15).
        Order: Quantize EACH query's update individually, then sum integers.
        Formula: w_new = w_old + sum( Q( step_i ) )
        """
        total_update_magnitude = 0.0
        weight_change = 0
        boundary_hits = 0
        attempted_updates = 0
        total_params = 0

        # Scale factor combines sigma scaling.
        # Note: 'alpha' here represents the learning rate (eta).
        # 'coeff' represents the sensitivity (mu_i) normalized by n (if applicable).
        inv_sigma = 1.0 / max(1e-8, float(sigma))

        # Generator for the stochastic rounding
        update_gen = torch.Generator(device=self.device)
        update_gen.manual_seed(update_seed)

        for name, module in self._iter_gptq_modules():
            dev = module.qweight.device
            w_bits = self._get_w_bits(module)
            limit_max = (1 << int(w_bits)) - 1

            # 1. Unpack current weights
            w_int = self._unpack_qweight(module.qweight, w_bits).to(torch.int32)
            
            # Accumulator for INTEGER updates
            delta_int_total = torch.zeros_like(w_int, dtype=torch.int32)

            for seed, coeff in per_seed_coeffs:
                seed_i = int(seed)
                
                # --- Q-RGE2 Reconstruction ---
                # Retrieve u_i using noise_seed, but quantize with round_seed_2
                round_seed_2 = seed_i + 67890 if seed_i >= 0 else -(abs(seed_i) + 67890)

                u_q2 = _get_stochastic_perturbation(
                    w_int.shape, dev, 
                    noise_seed=seed_i, 
                    noise_sigma=sigma, 
                    round_seed=round_seed_2, 
                    bits=w_bits
                ).to(torch.int32)

                # Boundary gating (based on w_int)
                w_candidate = w_int + u_q2
                boundary_mask = (w_candidate > limit_max) | (w_candidate < 0)
                u_eff = u_q2.masked_fill(boundary_mask, 0)
                
                # --- STRICT PAPER ORDER START ---
                
                # 1. Calculate partial step for THIS query in float
                # step_i = (coeff * u_{i,2}) * alpha * (1/sigma)
                partial_float_step = u_eff.to(torch.float32)
                partial_float_step.mul_(float(coeff) * float(alpha) * inv_sigma)

                # 2. Stochastic Rounding IMMEDIATELY [cite: 175, 180]
                # Q( step_i )
                partial_int_step = _stochastic_round_tensor(partial_float_step, update_gen).to(torch.int32)
                
                # 3. Accumulate Integer Update
                delta_int_total.add_(partial_int_step)
                
                # Metric tracking
                total_update_magnitude += partial_float_step.abs().mean().item()

                # --- STRICT PAPER ORDER END ---

            # Apply aggregated integer updates
            attempted_updates += (delta_int_total != 0).sum().item()

            # Anti-Windup & Boundary Check on the FINAL sum
            w_new = w_int + delta_int_total
            valid_mask = (w_new >= 0) & (w_new <= limit_max)

            boundary_hits += ((~valid_mask) & (delta_int_total != 0)).sum().item()

            # Zero out updates that would push weights out of bounds
            delta_final = torch.where(valid_mask, delta_int_total, torch.tensor(0, device=dev, dtype=torch.int32))
            w_final = w_int + delta_final

            # Pack and Write
            module.qweight.copy_(self._pack_qweight(w_final, module.qweight.dtype, w_bits))

            weight_change += (delta_final != 0).sum().item()
            total_params += w_int.numel()

        torch.cuda.empty_cache()
        ratio = weight_change / total_params if total_params > 0 else 0.0
        boundary_ratio = (boundary_hits / attempted_updates if attempted_updates > 0 else 0.0)
        
        print(
            f"QuZO Strict Update (Sum-Q). Sparsity: {ratio:.6f}, "
            f"BoundaryHit: {boundary_ratio:.6f}, "
            f"AvgPartialStep: {total_update_magnitude / len(per_seed_coeffs):.1e}"
        )
        return {
            "weight_change_ratio": ratio,
            "boundary_hits": int(boundary_hits),
            "attempted_updates": int(attempted_updates),
        }