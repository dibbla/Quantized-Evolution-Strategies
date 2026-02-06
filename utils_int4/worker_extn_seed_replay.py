from collections import deque
import torch
from utils_int4.worker_extn_full_precision import BaseESExtension, _get_stochastic_perturbation

class SeedReplayESExtension(BaseESExtension):
    """
    Scale-aware dual-evolution for GPTQ-Marlin int4 models with seed replay.
    - Perturbs packed integer codes with boundary-only gating
    - Replays random seeds for perturbation consistency across distributed workers
    """
    
    # The only difference from BaseESExtension is in the update step
    def _reconstruct_layer_residual(
        self, 
        module, 
        w_bits: int, 
        alpha: float, 
        decay: float, 
        current_w_int: torch.Tensor
    ) -> torch.Tensor:
        """
        Replays the history window to reconstruct the residual for a specific layer.
        """
        device = current_w_int.device
        # Start assuming 0 residual at the beginning of the window
        # (Approximation: errors from before the window have decayed to near zero)
        running_residual = torch.zeros_like(current_w_int, dtype=torch.float32)
        
        limit_max = (1 << int(w_bits)) - 1

        # We replay the exact logic of the update step
        for (step_coeffs, step_sigma) in self.update_history:
            scale_factor = 1.0 / max(1e-8, float(step_sigma))
            
            # 1. Re-calculate Gradient Estimation for this historical step
            grad_est = torch.zeros_like(current_w_int, dtype=torch.float32)
            
            for seed, coeff in step_coeffs:
                seed_i = int(seed)
                round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)

                # Re-generate the exact same noise pattern
                u_q = _get_stochastic_perturbation(
                    current_w_int.shape, device, 
                    noise_seed=seed_i, noise_sigma=step_sigma, round_seed=round_seed, bits=w_bits
                ).to(torch.int32)

                # Boundary gating (Approximation: We use CURRENT weights for boundary check)
                # In practice, weights change sparsely, so this mismatch is negligible.
                w_candidate = current_w_int + u_q
                boundary_mask = (w_candidate > limit_max) | (w_candidate < 0)
                u_eff = u_q.masked_fill(boundary_mask, 0)

                grad_est.add_(u_eff.to(torch.float32), alpha=float(coeff))

            grad_est.mul_(scale_factor)
            
            # 2. Replay the Residual Dynamics
            # float_step = grad * alpha + prev_resid * decay
            float_step = grad_est.mul(alpha).add(running_residual, alpha=decay)
            
            # 3. Simulate the Discrete Update
            delta_int = torch.round(float_step).to(torch.int32)
            
            # Apply Anti-Windup (using current weights as proxy)
            w_new = current_w_int + delta_int
            valid_mask = (w_new >= 0) & (w_new <= limit_max)
            delta_final = torch.where(valid_mask, delta_int, torch.tensor(0, device=device, dtype=torch.int32))
            
            # 4. Update the running residual for the next step
            # Residual = Float_Step - Applied_Int
            running_residual = float_step - delta_final.to(torch.float32)
            
            # Optional: Clamp to match the main logic if you use clamping there
            running_residual.clamp_(-1.5, 1.5)

        return running_residual

    @torch.no_grad()
    def apply_quzo_perturb_update(
        self,
        per_seed_coeffs: list,
        sigma: float = 0.1,
        alpha: float = 0.001,
        residual_decay: float = 0.9,
        window_size: int = 50,
    ):
        """
        Memory-optimized update. Reconstructs residuals on the fly.
        """
        if not hasattr(self, "update_history"):
            self.update_history = deque(maxlen=window_size)

        if residual_decay != 0.9 and window_size != 50:
            print(f"non-default decay/window_size used: decay={residual_decay}, window_size={window_size}")

        total_update_magnitude = 0.0
        weight_change = 0
        boundary_hits = 0
        attempted_updates = 0
        total_params = 0

        # We do NOT add the current step to history yet. We need to reconstruct
        # the residual state as it exists *before* this step.

        # Pre-calculate scale factor for the current step
        scale_factor = 1.0 / max(1e-8, float(sigma))

        # Iterate layer by layer
        for name, module in self._iter_gptq_modules():
            dev = module.qweight.device
            w_bits = self._get_w_bits(module)
            limit_max = (1 << int(w_bits)) - 1

            # 1. Unpack current weights
            w_int = self._unpack_qweight(module.qweight, w_bits).to(torch.int32)

            # 2. RECONSTRUCT RESIDUAL (The Magic Step)
            # This consumes compute but saves GBs of memory.
            # We reconstruct the residual based on the previous N steps.
            if len(self.update_history) > 0:
                prev_resid = self._reconstruct_layer_residual(
                    module, w_bits, alpha, residual_decay, w_int
                )
            else:
                # First step, random init or zero init
                prev_resid = (torch.rand_like(w_int, dtype=torch.float32) - 0.5)

            # 3. Calculate CURRENT Step Gradient
            grad_est = torch.zeros_like(w_int, dtype=torch.float32)
            for seed, coeff in per_seed_coeffs:
                seed_i = int(seed)
                round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)
                
                u_q = _get_stochastic_perturbation(
                    w_int.shape, dev, 
                    noise_seed=seed_i, noise_sigma=sigma, round_seed=round_seed, bits=w_bits
                ).to(torch.int32)

                w_candidate = w_int + u_q
                boundary_mask = (w_candidate > limit_max) | (w_candidate < 0)
                u_eff = u_q.masked_fill(boundary_mask, 0)
                grad_est.add_(u_eff.to(torch.float32), alpha=float(coeff))

            grad_est.mul_(scale_factor)

            # 4. Compute Update (Standard Logic)
            float_step = grad_est.mul(alpha)
            # Add the reconstructed residual
            float_step.add_(prev_resid, alpha=residual_decay)

            delta_int = torch.round(float_step).to(torch.int32)

            attempted_updates += (delta_int != 0).sum().item()

            # 5. Apply & Boundary Check
            w_new = w_int + delta_int
            valid_mask = (w_new >= 0) & (w_new <= limit_max)

            # Count how many elements would cross the code boundary (i.e., are blocked).
            # We only count attempted moves (delta != 0).
            boundary_hits += ((~valid_mask) & (delta_int != 0)).sum().item()

            delta_final = torch.where(valid_mask, delta_int, torch.tensor(0, device=dev, dtype=torch.int32))
            w_final = w_int + delta_final

            # 6. Pack and Write
            module.qweight.copy_(self._pack_qweight(w_final, module.qweight.dtype, w_bits))

            weight_change += (delta_final != 0).sum().item()
            total_params += w_int.numel()
            total_update_magnitude += float_step.abs().mean().item()
            
            # We discard `prev_resid` and `grad_est` here. 
            # Memory is freed immediately for the next layer.

        # 7. Update History for NEXT step
        # We append the current step's meta-data so it can be used for replay next time.
        self.update_history.append((per_seed_coeffs, sigma))

        torch.cuda.empty_cache()
        ratio = weight_change / total_params if total_params > 0 else 0.0
        boundary_ratio = (
            boundary_hits / attempted_updates if attempted_updates > 0 else 0.0
        )
        print(
            f"Stateless Update. Sparsity: {ratio:.6f}, "
            f"BoundaryHit: {boundary_ratio:.6f} ({int(boundary_hits)}/{int(attempted_updates)}), "
            f"AvgStep: {total_update_magnitude:.1e}"
        )
        return {
            "weight_change_ratio": ratio,
            "boundary_hits": int(boundary_hits),
            "boundary_hit_ratio": float(boundary_ratio),
            "attempted_updates": int(attempted_updates),
        }