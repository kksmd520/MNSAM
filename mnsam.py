import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from typing import List, Optional, Union, Tuple

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

class MNSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, beta1 = 0.9,momentum=0.9, nesterov=False,**kwargs):
        assert rho >= 0.0, "Invalid rho, should be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, beta1 = beta1,momentum=momentum,nesterov=nesterov, **kwargs)
        super(MNSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups,**kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        for group in self.param_groups:
            group.setdefault('momentum_buffer', {})

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            beta1 = group['beta1']
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]

                # Momentum update
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(p.grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    scale = group['rho'] / (torch.norm(buf, p=2) + 1e-12)

                # Save current parameters
                param_state['old_p'] = p.data.clone()

                # SAM gradient scale
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * buf * scale.to(p)

                p.add_(e_w)  # Update parameter

        if zero_grad:
            self.zero_grad()

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