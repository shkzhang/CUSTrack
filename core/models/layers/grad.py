# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/13
# @Time        : 18:25
# @Description :
import torch


class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """

    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None


def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)