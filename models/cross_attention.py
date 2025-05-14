import torch
from torch import nn
from torch.nn import functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            # 空洞卷积，通过调整dilation参数来捕获不同尺度的信息
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),  # 批量归一化
            nn.ReLU()  # ReLU激活函数
        ]
        super(ASPPConv, self).__init__(*modules)


class CrossAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, key_mask=None):
        N, C = kv.shape[1:]
        QN = q.shape[1]
        q = self.q(q).reshape([-1, QN, self.num_heads,
                               C // self.num_heads]).transpose(1, 2)
        q = q * self.scale
        k, v = self.kv(kv).reshape(
            [-1, N, 2, self.num_heads,
             C // self.num_heads]).permute(2, 0, 3, 1, 4)

        attn = q.matmul(k.transpose(2, 3))

        if key_mask is not None:
            attn = attn + key_mask.unsqueeze(1)

        attn = F.softmax(attn, -1)
        if not self.training:
            self.attn_map = attn
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose(1, 2).reshape((-1, QN, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiScaleCrossAttention(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        out_channels = out_channels  # 输出通道数
        modules = []
        self.aa = ASPPConv(in_channels, out_channels, 6)
        # 根据不同的膨胀率添加空洞卷积模块
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        self.convs = nn.ModuleList(modules)

        self.cr = CrossAttention(dim=out_channels,
                                 num_heads=num_heads,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,
                                 )

    def forward(self, x):
        for conv in self.convs:
            x = self.cr(x, conv(x.permute(0, 2, 1)).permute(0, 2, 1))
        return x

#test
# if __name__ == '__main__':
#     model = MultiScaleCrossAttention(in_channels=512,
#                                      out_channels=512,
#                                      atrous_rates=[6, 12, 18],
#                                      num_heads=8,
#                                      qkv_bias=False,
#                                      qk_scale=None,
#                                      attn_drop=0.0,
#                                      proj_drop=0.0, )
#     src = torch.rand(10, 32, 512)
#     out = model(src)
#     print(out.shape)
