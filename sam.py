import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()  #2范数
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  #less rho-t 即 local max sigma #类似标准化

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()  #保存
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)  #less sigma/sigma^3
                p.add_(e_w)  # climb to the local maximum "w + sigma/sigma^3"

        if zero_grad:
            self.zero_grad()  #梯度清零

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" 更新

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # 能计算梯度

        self.first_step(zero_grad=True)  #内变量
        closure()
        self.second_step()  #外变量

    def _grad_norm(self):  #梯度//w/*梯度 2范数
        shared_device = self.param_groups[0]["params"][0].device  # 指定 和参数相同的设备
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):  #用于保存和加载优化器的一个状态信息，state_dict() 方法里面保存了我们优化器的各种状态信息，可以通过 load_state_dict() 来导入这个状态信息，让优化器在这个基础上进行训练。
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups