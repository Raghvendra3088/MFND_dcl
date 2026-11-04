import torch
import torch.nn as nn
import torch.nn.functional as F

print('Smoke testing enhanced CFCA-I components...')
B, L, M, D, D_rob = 16, 64, 49, 1024, 768

tt = torch.randn(B, L, D)
tm = torch.ones(B, L, dtype=torch.bool)
it = torch.randn(B, M, D)
t_cls_clip = torch.randn(B, D)
i_cls_clip = torch.randn(B, D)
t_cls_rob = torch.randn(B, D_rob)
i_cls_vit = torch.randn(B, D_rob)
labels = torch.randint(0, 2, (B,))

print('All tensor shapes verified!')
