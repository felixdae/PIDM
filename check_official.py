import math
import torch as th

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

def official(q:th.Tensor, k:th.Tensor, v:th.Tensor, ch:int):
    # print(q.size(),k.size(),v.size())
    scale = 1 / math.sqrt(ch)
    return th.nn.functional.scaled_dot_product_attention(q,k,v,scale=scale)

if __name__ == '__main__':
    shape = [16, 256, 1024]
    q = th.randn(shape)
    k = th.randn(shape)
    v = th.randn(shape)

    ch = 256
    a = gold(q,k,v,ch)
    b = official(q.transpose(-2,-1),k.transpose(-2,-1),v.transpose(-2,-1),ch).transpose(-2,-1)

    print(a.size(), a[0,0,:5])
    print(b.size(), b[0,0,:5])
    print(th.max(th.abs(a-b)))
