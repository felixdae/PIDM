import math
import torch as th
import numpy as np
from xformers.ops import memory_efficient_attention

def gold(q:th.Tensor, k:th.Tensor, v:th.Tensor, ch:int):
    print("qkv shape", q.size(), k.size(), v.size())
    scale = 1 / math.sqrt(math.sqrt(ch))
    weight = th.einsum(
        "bct,bcs->bts", q * scale,
        k * scale)  # More stable with f16 than dividing afterwards
    weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
    print("qkv weight shape", weight.size())
    a = th.einsum("bts,bcs->bct", weight, v)
    return a

def xformers(q:th.Tensor, k:th.Tensor, v:th.Tensor, ch:int):
    scale = 1 / math.sqrt(ch)
    return memory_efficient_attention(q,k,v,scale=scale)


def relative_error(a:th.Tensor, b:th.Tensor):
    indcies = th.argmax((a-b).abs())
    max_index = np.unravel_index(indcies.cpu().numpy(), shape=a.size())
    v1 = a[max_index]
    v2 = b[max_index]
    return 2*(v1-v2).abs()/(v1.abs()+v2.abs())

if __name__ == '__main__':
    shape = [16, 256, 1024]
    q = th.randn(shape).cuda()
    k = th.randn(shape).cuda()
    v = th.randn(shape).cuda()

    ch = shape[-2]
    a = gold(q,k,v,ch)
    b = xformers(q.transpose(-2,-1).contiguous(),
                 k.transpose(-2,-1).contiguous(),
                 v.transpose(-2,-1).contiguous(),
                 ch).transpose(-2,-1)
    print(a.dtype, b.dtype)

    print(a.size(), a[0,0,:5])
    print(b.size(), b[0,0,:5])
    print(relative_error(a,b))
