from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=1e-3, betas=(0.9,0.95), eps=1e-8):
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "lambda": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lmbda = group["lambda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                beta1_t = state.get("beta1_t", beta1)
                beta2_t = state.get("beta2_t", beta2)
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = beta1*m + (1 - beta1)*grad
                v = beta2*v + (1 - beta2)*grad*grad
                alpha_t = alpha*math.sqrt(1-beta2_t)/(1-beta1_t)
                p.data -= alpha_t*m/(v.sqrt()+eps)
                p.data -= alpha*lmbda*p.data
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
                state["beta1_t"] = beta1_t*beta1
                state["beta2_t"] = beta2_t*beta2
        return loss
