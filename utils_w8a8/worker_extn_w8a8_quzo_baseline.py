import torch

from utils_w8a8.worker_extn_w8a8_full_precision import (
    W8A8ESExtension,
    _get_stochastic_perturbation_int8_signed,
)


def _stochastic_round_tensor(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Stochastic rounding for float tensors.

    Implements: floor(x) + Bernoulli(x - floor(x)).
    Works for negative values as well (unbiased).
    """

    floor_t = torch.floor(tensor)
    fractional_part = tensor - floor_t
    bernoulli_sample = torch.bernoulli(fractional_part, generator=generator)
    return floor_t + bernoulli_sample


class BaselineQuZOW8A8Extension(W8A8ESExtension):
    """QuZO baseline extension for W8A8 models.

    Mirrors `utils_int4.worker_extn_quzo_baseline.BaselineQuZOExtension` but targets
    signed int8 weight tensors (W8A8) instead of GPTQ-packed int4.

    Key QuZO behaviors:
    - Perturbation uses u_{i,1} = Q1(u_i) with (noise_seed=seed, round_seed_1=seed+12345)
    - Gradient estimation uses u_{i,2} = Q2(u_i) with SAME noise_seed but different
      rounding stream (round_seed_2=seed+67890)
    - Final update uses stochastic rounding driven by `update_seed`
    """

    @torch.no_grad()
    def apply_quzo_perturb_update(
        self,
        per_seed_coeffs: list,
        sigma: float = 0.1,
        alpha: float = 0.001,
        update_seed: int = 42,
        *args,
        **kwargs,
    ):
        total_update_magnitude = 0.0
        weight_change = 0
        boundary_hits = 0
        attempted_updates = 0
        total_params = 0

        scale_factor = 1.0 / max(1e-8, float(sigma))
        limit_min = -128
        limit_max = 127

        update_gen = torch.Generator(device=self.device)
        update_gen.manual_seed(int(update_seed))

        with torch.no_grad():
            for _, module in self._iter_int8_weight_modules():
                w = module.weight
                dev = w.device

                w_int = w.to(torch.int32)

                grad_est = torch.zeros_like(w_int, dtype=torch.float32)
                for seed, coeff in per_seed_coeffs:
                    seed_i = int(seed)

                    # QuZO: same continuous noise u_i, different rounding stream for u_{i,2}
                    round_seed_2 = seed_i + 67890 if seed_i >= 0 else -(abs(seed_i) + 67890)

                    u_q2 = _get_stochastic_perturbation_int8_signed(
                        w_int.shape,
                        dev,
                        noise_seed=seed_i,
                        noise_sigma=float(sigma),
                        round_seed=round_seed_2,
                        max_abs_delta=127,
                    ).to(torch.int32)

                    w_candidate = w_int + u_q2
                    boundary_mask = (w_candidate > limit_max) | (w_candidate < limit_min)
                    u_eff = u_q2.masked_fill(boundary_mask, 0)

                    grad_est.add_(u_eff.to(torch.float32), alpha=float(coeff))

                grad_est.mul_(scale_factor)

                float_step = grad_est.mul(float(alpha))

                # Stochastic rounding for the final update step.
                delta_int = _stochastic_round_tensor(float_step, update_gen).to(torch.int32)

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

        torch.cuda.empty_cache()
        ratio = weight_change / total_params if total_params > 0 else 0.0
        boundary_ratio = boundary_hits / attempted_updates if attempted_updates > 0 else 0.0

        print(
            f"QuZO Update. Sparsity: {ratio:.6f}, "
            f"BoundaryHit: {boundary_ratio:.6f} ({int(boundary_hits)}/{int(attempted_updates)}), "
            f"AvgStep: {total_update_magnitude:.1e}"
        )

        return {
            "weight_change_ratio": float(ratio),
            "boundary_hits": int(boundary_hits),
            "boundary_hit_ratio": float(boundary_ratio),
            "attempted_updates": int(attempted_updates),
        }
