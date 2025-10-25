# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "MSECSP",
    "LSKA",
    "DSKA",
    "MorphologicalPreprocess",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()
        # 可学习膨胀系数 (初始值针对横向优化)
        self.dilation_h = nn.Parameter(torch.tensor(3.0))  # 水平方向大膨胀
        self.dilation_v = nn.Parameter(torch.tensor(1.0))  # 垂直方向小膨胀

        # 横向分支增强
        self.conv_h = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, (1, 5),
                      padding=(0, 2), dilation=(1, 1)),  # 动态调整
            nn.BatchNorm2d(outplanes),
            nn.GELU()
        )

        # 纵向分支简化
        self.conv_v = nn.Conv2d(inplanes, outplanes, (3, 1), padding=(1, 0))

        # 特征融合
        self.fuse = nn.Conv2d(outplanes * 2, outplanes, 1)

    def forward(self, x):
        # 动态调整膨胀系数
        dilation_h = int(torch.clamp(self.dilation_h, 1, 8))
        dilation_v = int(torch.clamp(self.dilation_v, 1, 2))
        self.conv_h[0].dilation = (1, dilation_h)
        self.conv_h[0].padding = (0, dilation_h * 2)  # 保持输出尺寸

        # 动态调整垂直方向的膨胀系数
        self.conv_v.dilation = (dilation_v, 1)
        self.conv_v.padding = (dilation_v, 0)  # 保持输出尺寸

        # 特征提取
        x_h = self.conv_h(x)  # 水平特征
        x_v = self.conv_v(x)  # 垂直特征

        # 特征融合
        return self.fuse(torch.cat([x_h, x_v], dim=1))

class DynamicStripPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        if in_channels == 0:
            raise ValueError("DynamicStripPooling input channels cannot be zero")
        # 多尺度条带池化
        self.h_pool = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((None, 1)),
                nn.Conv2d(in_channels, in_channels, kernel_size=1)
            ),  # 全局
            nn.Sequential(
                nn.AvgPool2d((3, 1), stride=1, padding=(1, 0)),
                nn.Conv2d(in_channels, in_channels, kernel_size=1)
            )  # 局部
        ])
        self.v_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # 注意力门控
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            ChannelAttention(in_channels // 4),  # 新增通道注意力
            nn.Conv2d(in_channels // 4, 3, 1),  # 3种尺度权重
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 多尺度横向特征
        h_feats = [pool(x) for pool in self.h_pool]
        h_feats = [f.expand_as(x) for f in h_feats]

        # 纵向特征
        v_feat = self.v_pool(x).expand_as(x)

        # 动态融合
        weights = self.gate(x)  # [B,3,H,W]
        fused = weights[:, 0:1] * h_feats[0] + \
                weights[:, 1:2] * h_feats[1] + \
                weights[:, 2:3] * v_feat

        return x + fused


class LSKA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 使用可分离卷积来增加感受野
        self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=dim)

        # # 增加水平方向的感受野，减小垂直方向的核大小
        # self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), groups=dim,
        #                                 dilation=1)
        # self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=dim,
        #                                 dilation=1)

        # 增加水平方向的感受野，减小垂直方向的核大小
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 31), stride=(1, 1), padding=(0, 15), groups=dim,
                                        dilation=1)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=dim,
                                        dilation=1)

        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # # 显著增加水平方向的感受野
        # self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 63), stride=(1, 1), padding=(0, 31), groups=dim,
        #                                 dilation=1)
        # self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=dim,
        #                                 dilation=1)
        #
        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        u = x.clone()

        # 水平和垂直方向的注意力
        attn_h = self.conv0h(x)
        attn_v = self.conv0v(x)
        attn = attn_h + attn_v

        # 增加感受野
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)

        # 全局平均池化
        attn = self.global_avg_pool(attn)
        attn = F.relu(attn)
        attn = attn.expand_as(u)

        # 最后的注意力卷积
        attn = self.conv1(attn)
        return u * attn

class CSKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用非对称卷积核
        self.conv_h = nn.Sequential(
            nn.Conv2d(dim, dim, (3, 3), padding=(1, 1), groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        # 添加坐标注意力
        self.ca = ChannelAttention(dim)

    def forward(self, x):
        x = self.conv_h(x)
        x = self.ca(x) * x  # 增强通道感知
        return x


class DSKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 极宽水平卷积核 + 动态膨胀系数
        self.conv_h = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1)),  # 增强特征交互
            nn.GELU()
        )
        # 轻量化垂直卷积
        self.conv_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1)),  # 增强特征交互
            nn.GELU()
        )
        # 动态门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h_feat = self.conv_h(x)  # [B,C,H,W]
        v_feat = self.conv_v(x)

        # 动态权重融合
        gate = self.gate(x)  # [B,2,1,1]
        fused = gate[:, 0:1] * h_feat + gate[:, 1:2] * v_feat

        return x * fused


class MSECSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super(MSECSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
        self.dsp = DynamicStripPooling(c_)
        self.dska = DSKA(c_ * 2)
        # self.dska = CSKA(c_ * 2)
        self.residual = nn.Sequential(
            *[Conv(c_, c_, 3, 1) for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x))) + x
        x2 = self.m(x1)
        x3 = self.m(x2)
        x5 = self.dsp(x1)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x5), 1)))
        y2 = self.cv2(x)
        return self.cv7(self.dska(torch.cat((y1, y2), dim=1)))

# class MorphologicalPreprocess(nn.Module):
#     def __init__(self, in_channels, out_channels, target_length=960):
#         super().__init__()
#         self.target_length = target_length
#         # 计算形态学操作生成的通道数
#         morph_channels = self.calculate_morphological_channels(in_channels)
#         # 形态学特征通道数设置为与输入相同
#         self.morph_conv = nn.Conv2d(morph_channels, in_channels, kernel_size=3, padding=1)
#
#         # 通道注意力权重生成
#         self.attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 3, 1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // 3, in_channels, 1),
#             nn.Sigmoid()
#         )
#
#         # 最终融合卷积
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#
#     # def calculate_morphological_channels(self, in_channels):
#     #     dummy_input = torch.zeros(1, in_channels, 3, 3)
#     #     features = self.morphological_operations(dummy_input)
#     #     return features.shape[1] // in_channels
#     def calculate_morphological_channels(self, in_channels):
#         dummy_input = torch.zeros(1, in_channels, 3, 3)
#         features = self.morphological_operations(dummy_input)
#         return features.shape[1]
#
#     def morphological_operations(self, x):
#         # 输入形状: [B, 3, H, W]
#         B, C, H, W = x.shape
#
#         # 横向处理（3通道）
#         horizontal_kernel = (1, 7)  # OpenCV (15,1) → PyTorch (H, W)
#         h_tophat = F.max_pool2d(x, kernel_size=horizontal_kernel, stride=1, padding=(0, 3)) - x
#         h_gradient = F.max_pool2d(x, kernel_size=horizontal_kernel, stride=1, padding=(0, 3)) - \
#                      F.max_pool2d(x, kernel_size=horizontal_kernel, stride=1, padding=(0, 3))
#         h_blackhat = x - F.max_pool2d(x, kernel_size=horizontal_kernel, stride=1, padding=(0, 3))
#         # h_processed = 0.7 * h_blackhat + 0.3 * h_gradient
#         h_processed = torch.max(h_tophat, h_blackhat)
#
#         # 纵向处理（3通道）
#         vertical_kernel = (3, 1)  # OpenCV (1,7) → PyTorch (H, W)
#         v_tophat = F.max_pool2d(x, kernel_size=vertical_kernel, stride=1, padding=(1, 0)) - x
#         v_blackhat = x - F.max_pool2d(x, kernel_size=vertical_kernel, stride=1, padding=(1, 0))
#         v_processed = torch.max(v_tophat, v_blackhat)
#
#         # 点状处理（3通道）
#         kernel_s = (3, 3)
#         kernel_m = (5, 5)
#         open_s = F.max_pool2d(F.max_pool2d(x, kernel_size=kernel_s, stride=1, padding=1), kernel_size=kernel_s,
#                               stride=1, padding=1)
#         open_m = F.max_pool2d(F.max_pool2d(x, kernel_size=kernel_m, stride=1, padding=2), kernel_size=kernel_m,
#                               stride=1, padding=2)
#         dots = open_s - open_m
#         # dots = open_s
#
#         # 融合特征（9通道）
#         morph_features = torch.cat([h_processed, v_processed, dots], dim=1)
#         # morph_features = torch.cat([h_blackhat, v_blackhat, dots], dim=1)
#
#         # 拼接原始图像（3通道）和形态学特征（9通道）→ 总12通道
#         x = torch.cat([x, morph_features], dim=1)
#
#         return morph_features
#
#     def forward(self, x):
#         # 形态学特征提取
#         morph_feat = self.morph_conv(self.morphological_operations(x))
#
#         # 注意力权重
#         attn_weights = self.attn(morph_feat)
#
#         # x = torch.cat((x, morph_feat), dim=1)
#
#         # 加权融合
#         # fused = x * (1 - attn_weights) + morph_feat * attn_weights
#         # 使用归一化或归一化融合
#         # fused = torch.cat([x * (1 - attn_weights), morph_feat * attn_weights], dim=1)
#         fused = x + morph_feat * attn_weights
#
#         # 特征压缩
#         # fused = fused.permute(0, 2, 1, 3)  # (B, H, C, W) -> 转换为 H × C × W
#         # fused_compressed = self.adaptive_downsample(fused, self.target_length)  # 添加 target_length 参数
#         # fused_compressed = fused_compressed.permute(0, 2, 1, 3)  # 恢复为 (B, C, H, W_compressed)
#
#         # 使用 FrequencyAwareDownsampler 替代自定义下采样
#         # x_down = self.downsample(fused)  # [N,C,H,960]
#
#         # 最终融合卷积
#         return self.fusion_conv(fused)

class MorphologicalPreprocess(nn.Module):
    def __init__(self, in_channels, out_channels, target_length=960):
        super().__init__()
        self.target_length = target_length
        # 计算形态学操作生成的通道数
        self.morph_channels = self.calculate_morphological_channels(in_channels)
        # 形态学特征通道数设置为与输入相同
        self.morph_conv = nn.Conv2d(in_channels*self.morph_channels, in_channels, kernel_size=3, padding=1)

        # 通道注意力权重生成
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 3, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 3, in_channels, 1),
            nn.Sigmoid()
        )

        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def calculate_morphological_channels(self, in_channels):
        dummy_input = torch.zeros(1, in_channels, 3, 3)
        features = self.morphological_operations(dummy_input)
        return features.shape[1] // in_channels

    def morphological_operations(self, x):
        # 输入形状: [B, 3, H, W]
        B, C, H, W = x.shape

        # 亮度标准化
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        # 横向处理（3通道）
        horizontal_kernel = (1, 5)
        h_erosion = -F.max_pool2d(-x, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
        h_opening = F.max_pool2d(h_erosion, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
        h_tophat = x - h_opening
        h_dilation = F.max_pool2d(x, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
        h_closing = -F.max_pool2d(-h_dilation, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
        h_blackhat = h_closing - x
        # h_processed = torch.cat([h_tophat, h_blackhat, (h_tophat + h_blackhat) / 2], dim=1)  # 3通道
        h_processed = (h_tophat + h_blackhat) / 2  # 3通道

        # 纵向处理（3通道）
        vertical_kernel = (5, 1)
        v_erosion = -F.max_pool2d(-x, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
        v_opening = F.max_pool2d(v_erosion, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
        v_tophat = x - v_opening
        v_dilation = F.max_pool2d(x, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
        v_closing = -F.max_pool2d(-v_dilation, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
        v_blackhat = v_closing - x
        # v_processed = torch.cat([v_tophat, v_blackhat, (v_tophat + v_blackhat) / 2], dim=1)  # 3通道
        v_processed = (v_tophat + v_blackhat) / 2  # 3通道

        # 点状处理（3通道）
        kernel_s = (3, 3)
        kernel_m = (5, 5)
        open_s = F.avg_pool2d(F.max_pool2d(x, kernel_s, stride=1, padding=1), kernel_s, stride=1, padding=1)
        open_m = F.avg_pool2d(F.max_pool2d(x, kernel_m, stride=1, padding=2), kernel_m, stride=1, padding=2)
        # dots = torch.cat([open_s - open_m, open_s, open_m], dim=1)  # 3通道
        dots = open_s - open_m # 3通道

        # 融合特征（9通道）
        morph_features = torch.cat([h_processed, v_processed, dots], dim=1)
        # morph_features = torch.cat([h_gradient, v_processed, dots], dim=1)

        # 拼接原始图像（3通道）和形态学特征（9通道）→ 总12通道
        # x = torch.cat([x, morph_features], dim=1)

        return morph_features

    def forward(self, x):
        # 形态学特征提取
        morph_feat = self.morph_conv(self.morphological_operations(x))

        # 注意力权重
        attn_weights = self.attn(morph_feat)

        # x = torch.cat((x, morph_feat), dim=1)

        # 加权融合
        # fused = x * (1 - attn_weights) + morph_feat * attn_weights
        # 使用归一化或归一化融合
        # fused = torch.cat([x * (1 - attn_weights), morph_feat * attn_weights], dim=1)
        fused = x + morph_feat * attn_weights


        # 最终融合卷积
        return self.fusion_conv(fused)

# class MorphologicalPreprocess(nn.Module):
#     def __init__(self, in_channels, out_channels, target_length=960):
#         super().__init__()
#         self.target_length = target_length
#         # 计算形态学操作生成的通道数
#         self.morph_channels = self.calculate_morphological_channels(in_channels)
#         # 形态学特征通道数设置为与输入相同
#         self.morph_conv = nn.Conv2d(in_channels * self.morph_channels, in_channels, kernel_size=3, padding=1)
#
#         # 通道注意力权重生成
#         self.channel_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 3, 1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // 3, in_channels, 1),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力权重生成
#         self.spatial_attn = nn.Sequential(
#             nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
#
#         # 最终融合卷积
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#
#     def calculate_morphological_channels(self, in_channels):
#         dummy_input = torch.zeros(1, in_channels, 3, 3)
#         features = self.morphological_operations(dummy_input)
#         return features.shape[1] // in_channels
#
#     def morphological_operations(self, x):
#         # 输入形状: [B, 3, H, W]
#         B, C, H, W = x.shape
#
#         # 亮度标准化
#         x = (x - x.min()) / (x.max() - x.min() + 1e-6)
#
#         # 横向处理（3通道）
#         horizontal_kernel = (1, 5)
#         h_erosion = -F.max_pool2d(-x, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
#         h_opening = F.max_pool2d(h_erosion, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
#         h_tophat = x - h_opening
#         h_dilation = F.max_pool2d(x, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
#         h_closing = -F.max_pool2d(-h_dilation, kernel_size=horizontal_kernel, stride=1, padding=(0, 2))
#         h_blackhat = h_closing - x
#         h_processed = (h_tophat + h_blackhat) / 2  # 3通道
#
#         # 纵向处理（3通道）
#         vertical_kernel = (5, 1)
#         v_erosion = -F.max_pool2d(-x, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
#         v_opening = F.max_pool2d(v_erosion, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
#         v_tophat = x - v_opening
#         v_dilation = F.max_pool2d(x, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
#         v_closing = -F.max_pool2d(-v_dilation, kernel_size=vertical_kernel, stride=1, padding=(2, 0))
#         v_blackhat = v_closing - x
#         v_processed = (v_tophat + v_blackhat) / 2  # 3通道
#
#         # 点状处理（3通道）
#         kernel_s = (3, 3)
#         kernel_m = (5, 5)
#         open_s = F.avg_pool2d(F.max_pool2d(x, kernel_s, stride=1, padding=1), kernel_s, stride=1, padding=1)
#         open_m = F.avg_pool2d(F.max_pool2d(x, kernel_m, stride=1, padding=2), kernel_m, stride=1, padding=2)
#         dots = open_s - open_m  # 3通道
#
#         # 融合特征（9通道）
#         morph_features = torch.cat([h_processed, v_processed, dots], dim=1)
#
#         return morph_features
#
#     def forward(self, x):
#         # 形态学特征提取
#         morph_feat = self.morph_conv(self.morphological_operations(x))
#
#         # 通道注意力权重
#         channel_attn_weights = self.channel_attn(morph_feat)
#
#         # 空间注意力权重
#         spatial_attn_weights = self.spatial_attn(morph_feat)
#
#         # 合并注意力权重
#         attn_weights = channel_attn_weights * spatial_attn_weights
#
#         # 加权融合
#         fused = morph_feat * attn_weights
#
#         # 最终融合卷积
#         return self.fusion_conv(fused)


class LightMorphologicalPreprocess(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, target_length=960):
        super().__init__()
        self.target_length = target_length
        self.in_ch = in_channels

        # 1. 动态计算形态学通道倍数（封装逻辑保留）
        self.morph_ch = self._calc_morph_channel_multiple()
        # 2. 形态学特征降维：1×1卷积（轻量化核心，替代3×3）
        self.morph_conv = nn.Conv2d(in_channels * self.morph_ch, in_channels,
                                    kernel_size=1, bias=False)

        # 3. 通道注意力：保留原“压缩-激活”原理，封装为独立模块
        mid_ch = max(in_channels // 3, 1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局通道信息聚合
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 4. 补回轻量化空间注意力（原原理保留，仅精简卷积核）
        # 原7×7→3×3，计算量降为(3²)/(7²)=1/5.4，padding=1保持空间尺寸一致
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),  # 单通道空间权重
            nn.Sigmoid()  # 空间注意力归一化
        )

        # 5. 最终融合：1×1卷积+BN+ReLU（轻量化逻辑保留）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _calc_morph_channel_multiple(self):
        """封装动态通道倍数计算，加no_grad减少初始化开销"""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_ch, 3, 3)
            morph_feat = self._morphological_operations(dummy)
        return morph_feat.shape[1] // self.in_ch

    def _morph_single_branch(self, x, kernel, padding):
        """封装横向/纵向形态学逻辑，等价原“腐蚀-开运算-顶帽 + 膨胀-闭运算-黑帽”"""
        opening = F.max_pool2d(-F.max_pool2d(-x, kernel, 1, padding), kernel, 1, padding)
        closing = -F.max_pool2d(-F.max_pool2d(x, kernel, 1, padding), kernel, 1, padding)
        return (closing - opening) / 2  # 等价原(h_tophat + h_blackhat)/2

    def _morphological_operations(self, x):
        """封装完整形态学操作，保留原三分支逻辑（横向+纵向+点状）"""
        # 亮度标准化
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)

        # 三分支特征提取
        h_feat = self._morph_single_branch(x_norm, (1, 5), (0, 2))  # 横向
        v_feat = self._morph_single_branch(x_norm, (5, 1), (2, 0))  # 纵向
        # 点状分支（原逻辑保留）
        open_s = F.avg_pool2d(F.max_pool2d(x_norm, (3, 3), 1, 1), (3, 3), 1, 1)
        open_m = F.avg_pool2d(F.max_pool2d(x_norm, (5, 5), 1, 2), (5, 5), 1, 2)
        dot_feat = open_s - open_m

        # 融合三分支（总通道数=in_ch×3，与原一致）
        return torch.cat([h_feat, v_feat, dot_feat], dim=1)

    def forward(self, x):
        """前向传播：完全对齐原模块流程（形态学→双注意力→融合）"""
        # 1. 形态学特征提取+降维
        morph_raw = self._morphological_operations(x)
        morph_feat = self.morph_conv(morph_raw)  # 降为in_ch通道，匹配注意力输入

        # 2. 双注意力联合加权（原核心逻辑：通道权重×空间权重）
        channel_w = self.channel_attn(morph_feat)  # [B, in_ch, 1, 1]
        spatial_w = self.spatial_attn(morph_feat)  # [B, 1, H, W]
        joint_attn_w = channel_w * spatial_w  # 广播后逐元素相乘，联合加权

        # 3. 注意力加权+残差融合（保留原输入特征，增强鲁棒性）
        fused = x + morph_feat * joint_attn_w

        # 4. 最终通道映射输出
        return self.fusion_conv(fused)



class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
        # self.lska = LSKA(c_ * 2)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))