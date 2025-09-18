import math


def get_learning_rate(
    t: int, lr_max: float, lr_min: float, t_w: int, t_c: int
) -> float:
    if t < t_w:
        return t / t_w * lr_max
    if t <= t_c:
        return lr_min + 0.5 * (1.0 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (
            lr_max - lr_min
        )
    return lr_min
