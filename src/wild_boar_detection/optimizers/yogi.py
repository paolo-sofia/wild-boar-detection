from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

__all__ = ("Yogi",)

MIN = 0.0
MAX = 1.0


class Yogi(Optimizer):
    """Implements Yogi Optimizer Algorithm.

    It has been proposed in `Adaptive methods for Nonconvex Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 0.001)
        initial_accumulator: initial values for first and
            second moments (default: 1e-6)
        weight_decay: weight decay (L2 penalty) (default: 0)

    __ https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization  # noqa

    Note:
        Reference code: https://github.com/4rtemi5/Yogi-Optimizer_Keras
    """

    def __init__(
        self: Yogi,
        params: Params,
        lr: float = 1e-2,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        if lr <= MIN:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < MIN:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not MIN <= betas[0] < MAX:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not MIN <= betas[1] < MAX:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "initial_accumulator": initial_accumulator,
            "weight_decay": weight_decay,
        }
        super(Yogi, self).__init__(params, defaults)

    def step(self: Yogi, closure: OptLossClosure = None) -> OptFloat:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients, " "please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                # Followed from official implementation in tensorflow addons:
                # https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/yogi.py#L118
                # For more details refer to the discussion:
                # https://github.com/jettify/pytorch-optimizer/issues/77
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = nn.init.constant_(
                        torch.empty_like(p.data, memory_format=torch.preserve_format),
                        group["initial_accumulator"],
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = nn.init.constant_(
                        torch.empty_like(p.data, memory_format=torch.preserve_format),
                        group["initial_accumulator"],
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_squared = grad.mul(grad)

                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                    value=-(1 - beta2),
                )

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
