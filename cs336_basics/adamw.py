from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

from cs336_basics import gradient_clipping
from cs336_basics import lr_cosine_schedule
from cs336_basics import timer


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        extra_defaults=None,
    ):
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "lambda": weight_decay,
        }
        if extra_defaults is not None:
            defaults.update(extra_defaults)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lmbda = group["lambda"]
            gradient_scale = group.get("gradient_scale", None)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                if "alpha" in state:
                    alpha = state.get("alpha")
                t = state.get("t", 0)
                if "m" in state:
                    m = state.get("m")
                else:
                    m = torch.zeros_like(p.data)
                if "v" in state:
                    v = state.get("v")
                else:
                    v = torch.zeros_like(p.data)
                beta1_t = state.get("beta1_t", beta1)
                beta2_t = state.get("beta2_t", beta2)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                if gradient_scale is not None:
                    grad = timer.measure(
                        "adamw_gradient_scale", lambda: gradient_scale * grad
                    )
                m = timer.measure(
                    "adamw_update_m", lambda: beta1 * m + (1 - beta1) * grad
                )
                v = timer.measure(
                    "adamw_update_v",
                    lambda: beta2 * v + (1 - beta2) * grad.pow(2),
                )
                alpha_t = timer.measure(
                    "adamw_alpha_t",
                    lambda: alpha * math.sqrt(1 - beta2_t) / (1 - beta1_t),
                )
                p.data = timer.measure(
                    "adamw_update_params",
                    lambda: p.data - alpha_t * m / (v.sqrt() + eps),
                )
                p.data = timer.measure(
                    "adamw_weight_decay", lambda: p.data - alpha * lmbda * p.data
                )
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
                state["beta1_t"] = beta1_t * beta1
                state["beta2_t"] = beta2_t * beta2
        return loss


class AdamWextra(AdamW):
    def __init__(
        self,
        params,
        lr_max: float,
        lr_min: float,
        t_w: int,
        t_c: int,
        max_gradient_norm: float,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        device=None,
    ):
        super().__init__(
            params,
            0.0,
            weight_decay,
            betas,
            eps,
            {
                "alpha_max": lr_max,
                "alpha_min": lr_min,
                "t_w": t_w,
                "t_c": t_c,
                "max_gradient_norm": torch.as_tensor(
                    [max_gradient_norm], device=device
                ),
                "gradient_clipping_eps": torch.as_tensor([1e-6], device=device),
            },
        )

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            alpha_max = group["alpha_max"]
            alpha_min = group["alpha_min"]
            t_w = group["t_w"]
            t_c = group["t_c"]
            max_gradient_norm = group["max_gradient_norm"]
            gradient_clipping_eps = group["gradient_clipping_eps"]
            norm = timer.measure(
                "gradient_clipping_compute_norm",
                lambda: gradient_clipping.compute_norm(group["params"]),
            )
            scale = timer.measure(
                "gradient_clipping_compute_scale",
                lambda: gradient_clipping.compute_scale(
                    norm, max_gradient_norm, gradient_clipping_eps
                ),
            )
            group["gradient_scale"] = scale
            for p in group["params"]:
                state = self.state[p]
                t = state.get("t", 0)
                state["alpha"] = timer.measure(
                    "adamw_lr_schedule",
                    lambda: lr_cosine_schedule.get_learning_rate(
                        t, alpha_max, alpha_min, t_w, t_c
                    ),
                )
        return super().step(closure)
