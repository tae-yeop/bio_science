import torch

l = (
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D",
)
l2 = [3] * 3

a = list(reversed(l))
print(type(a), a)

a = reversed(l2)
print(type(a), a)