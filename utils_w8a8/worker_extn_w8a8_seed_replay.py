from __future__ import annotations

from collections import deque

import torch

from utils_w8a8.worker_extn_w8a8_full_precision import (
    W8A8ESExtension,
    _get_stochastic_perturbation_int8_signed,
)


class W8A8SeedReplayESExtension(W8A8ESExtension):
    """Seed-replay variant of `W8A8ESExtension`.

    This mirrors `utils_int4.worker_extn_seed_replay.SeedReplayESExtension`:
    - Avoids storing per-layer residual tensors (saves GPU/CPU memory)
    - Reconstructs the residual state on the fly by replaying recent (seed, sigma) history

    All other behaviors (perturb/restore/broadcast/iteration of modules) are inherited.
    """

    def _reconstruct_layer_residual(
        self,
        *,
        alpha: float,
        decay: float,
        current_w_int: torch.Tensor,
        limit_min: int,
        limit_max: int,
    ) -> torch.Tensor:
        device = current_w_int.device
        running_residual = torch.zeros_like(current_w_int, dtype=torch.float32)

        for (step_coeffs, step_sigma) in self.update_history:
            scale_factor = 1.0 / max(1e-8, float(step_sigma))

            grad_est = torch.zeros_like(current_w_int, dtype=torch.float32)
            for seed, coeff in step_coeffs:
                seed_i = int(seed)
                round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)

                u_q = _get_stochastic_perturbation_int8_signed(
                    current_w_int.shape,
                    device,
                    noise_seed=seed_i,
                    noise_sigma=float(step_sigma),
                    round_seed=round_seed,
                    max_abs_delta=127,
                ).to(torch.int32)

                w_candidate = current_w_int + u_q
                boundary_mask = (w_candidate > limit_max) | (w_candidate < limit_min)
                u_eff = u_q.masked_fill(boundary_mask, 0)

                grad_est.add_(u_eff.to(torch.float32), alpha=float(coeff))

            grad_est.mul_(scale_factor)

            float_step = grad_est.mul(float(alpha)).add(running_residual, alpha=float(decay))
            delta_int = torch.round(float_step).to(torch.int32)

            w_new = current_w_int + delta_int
            valid_mask = (w_new >= limit_min) & (w_new <= limit_max)
            delta_final = torch.where(
                valid_mask,
                delta_int,
                torch.tensor(0, device=device, dtype=torch.int32),
            )

            running_residual = float_step - delta_final.to(torch.float32)
            running_residual.clamp_(-1.5, 1.5)

        return running_residual

    @torch.no_grad()
    def apply_quzo_perturb_update(
        self,
        per_seed_coeffs: list,
        sigma: float = 0.1,
        alpha: float = 0.001,
        keep_residual: bool = True,
        residual_decay: float = 0.9,
    ):
        """Apply QuZO-style discrete ES update, with optional seed-replay residuals.

        Signature matches `W8A8ESExtension.apply_quzo_perturb_update` so the driver code
        can switch worker classes without changes.
        """
        print("W8A8SeedReplayESExtension: apply_quzo_perturb_update called with keep_residual="
              f"{keep_residual}")

        if not keep_residual:
            return super().apply_quzo_perturb_update(
                per_seed_coeffs,
                sigma=float(sigma),
                alpha=float(alpha),
                keep_residual=False,
                residual_decay=float(residual_decay),
            )

        if not hasattr(self, "update_history"):
            self.update_history = deque(maxlen=50)

        total_update_magnitude = 0.0
        weight_change = 0
        boundary_hits = 0
        attempted_updates = 0
        total_params = 0

        scale_factor = 1.0 / max(1e-8, float(sigma))
        limit_min = -128
        limit_max = 127

        with torch.no_grad():
            for _, module in self._iter_int8_weight_modules():
                w = module.weight
                dev = w.device

                w_int = w.to(torch.int32)

                if len(self.update_history) > 0:
                    prev_resid = self._reconstruct_layer_residual(
                        alpha=float(alpha),
                        decay=float(residual_decay),
                        current_w_int=w_int,
                        limit_min=limit_min,
                        limit_max=limit_max,
                    )
                else:
                    prev_resid = torch.rand_like(w_int, dtype=torch.float32) - 0.5

                grad_est = torch.zeros_like(w_int, dtype=torch.float32)
                for seed, coeff in per_seed_coeffs:
                    seed_i = int(seed)
                    round_seed = seed_i + 12345 if seed_i >= 0 else -(abs(seed_i) + 12345)

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

                float_step = grad_est.mul(float(alpha))
                float_step.add_(prev_resid, alpha=float(residual_decay))

                delta_int = torch.round(float_step).to(torch.int32)

                attempted_updates += (delta_int != 0).sum().item()

                w_new = w_int + delta_int
                valid_mask = (w_new >= limit_min) & (w_new <= limit_max)

                boundary_hits += ((~valid_mask) & (delta_int != 0)).sum().item()

                delta_final = torch.where(
                    valid_mask,
                    delta_int,
                    torch.tensor(0, device=dev, dtype=torch.int32),
                )
                w_final = w_int + delta_final

                module.weight.copy_(w_final.to(torch.int8))

                weight_change += (delta_final != 0).sum().item()
                total_params += w_int.numel()
                total_update_magnitude += float_step.abs().mean().item()

        self.update_history.append((per_seed_coeffs, float(sigma)))

        torch.cuda.empty_cache()
        ratio = weight_change / total_params if total_params > 0 else 0.0
        boundary_ratio = boundary_hits / attempted_updates if attempted_updates > 0 else 0.0
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
