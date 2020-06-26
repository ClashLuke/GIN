import torch
from memcnn import AdditiveCoupling, InvertibleModuleWrapper
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, ConvTranspose2d, Linear, Module


@torch.jit.script
def mish(fn_input):
    return torch.nn.functional.softplus(fn_input).tanh().mul(fn_input)


activation = mish


def arange_like(dim: int, tensor: torch.Tensor) -> torch.Tensor:
    out = torch.arange(1,
                       tensor.size(dim) + 1,
                       device=tensor.device,
                       dtype=tensor.dtype)
    dims = [1] * tensor.ndim
    dims[dim] = -1
    out = out.view(*dims)
    target = list(tensor.size())
    target[1] = -1
    out = out.expand(target)
    return out


class Activate(Module):
    def __init__(self, features: int):
        super(Activate, self).__init__()
        self.norm = BatchNorm2d(features)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return activation(self.norm(fn_input))


class SeparableConvolution(Module):
    def __init__(self, in_features: int, out_features: int, stride=1, dilation=1,
                 transpose=False, kernel_size=3):
        super(SeparableConvolution, self).__init__()
        kernel_size += transpose
        conv = ConvTranspose2d if transpose else Conv2d
        self.depth = conv(in_features, in_features, kernel_size, groups=in_features,
                          dilation=dilation, stride=stride, bias=False,
                          padding=dilation * ((kernel_size - 1) // 2))
        self.mid_norm = BatchNorm2d(in_features)
        self.point = Conv2d(in_features, out_features, 1)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        return self.point(activation(self.mid_norm(self.depth(fn_input))))


class PositionalSeparableConvolution(SeparableConvolution):
    def __init__(self, in_features: int, out_features: int, stride=1, dilation=1,
                 transpose=False, kernel_size=3):
        super(PositionalSeparableConvolution, self).__init__(4 + in_features,
                                                             out_features,
                                                             stride, dilation,
                                                             transpose, kernel_size)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        batch, features, width, height = fn_input.size()
        width_range = arange_like(2, fn_input)
        height_range = arange_like(3, fn_input)
        div_width_range = 2 * width_range / width - 1
        div_height_range = 2 * height_range / height - 1
        fn_input = torch.cat([width_range,
                              div_width_range,
                              height_range,
                              div_height_range,
                              fn_input], 1)
        return super().forward(fn_input)


class BasicModule(Module):
    def __init__(self, in_features, out_features, dilation, transpose, kernel_size,
                 heads, top_norm=True, stride=1):
        super(BasicModule, self).__init__()
        self.top_norm = BatchNorm2d(in_features) if top_norm else None
        self.top_conv = PositionalSeparableConvolution(in_features, in_features,
                                                       dilation=1, transpose=False,
                                                       kernel_size=3)
        self.mid_norm = BatchNorm2d(in_features)
        self.bot_conv = PositionalSeparableConvolution(in_features, out_features,
                                                       dilation=dilation,
                                                       stride=stride,
                                                       transpose=transpose,
                                                       kernel_size=kernel_size)

        self.excite = bool(heads)

        if self.excite:
            self.att_conv = PositionalSeparableConvolution(in_features, heads)
            self.att_norm_0 = BatchNorm1d(heads * in_features)
            self.att_proj_0 = Linear(heads * in_features, in_features, bias=False)
            self.att_norm_1 = BatchNorm1d(in_features)
            self.att_proj_1 = Linear(in_features, out_features)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        batch, in_features, height, width = fn_input.size()

        if self.top_norm is not None:
            fn_input = self.top_norm(fn_input)
        out = self.top_conv(fn_input)
        out = activation(self.mid_norm(out))
        out = self.bot_conv(out)

        if self.excite:
            att = self.att_conv(fn_input)
            att = torch.nn.functional.softmax(att.view(batch, 1, -1, height * width),
                                              -1)
            att = fn_input.view(batch, -1, 1, height * width) * att
            att = att.sum(-1)
            att = att.view(batch, -1)
            att = self.att_norm_0(att)
            att = self.att_proj_0(att)
            att = self.att_norm_1(att)
            att = self.att_proj_1(att)
            att = att.tanh()
            att = att.view(*att.size(), 1, 1)

            out = att * out

        return out


class ZeroPad(Module):
    def __init__(self, dim, amount):
        super(ZeroPad, self).__init__()
        self.dim = dim
        self.amount = amount

    def forward(self, fn_input: torch.Tensor, pad=0) -> torch.Tensor:
        size = list(fn_input.size())
        size[self.dim] = self.amount
        pad = torch.zeros(size, device=fn_input.device, requires_grad=False) + pad
        out = torch.cat([fn_input, pad], self.dim)
        return out

    def inverse(self, fn_input: torch.Tensor) -> torch.Tensor:
        return fn_input[:, 0:3, :, :]

    def __str__(self):
        return f'{type(self).__name__}({self.dim}, {self.amount})'

    def __repr__(self):
        return str(self)


def reversible_module(inf, outf, *args, stride=1, revnet=False, **kwargs):
    if revnet and stride == 1:
        inf //= 2
        outf //= 2
    module = BasicModule(inf, outf, *args, **kwargs, stride=stride)
    if revnet and stride == 1:
        return InvertibleModuleWrapper(AdditiveCoupling(module))
    return module


def linear_dilated_model(in_features, hidden_features, out_features=None, depth=4,
                         transpose=False, kernel_size=3, end=None, stride=1,
                         blocks_per_level=1, heads=12, revnet=False, wrap=True, target_coverage=1):
    linear_block_count = 1
    block_size = depth // linear_block_count

    while block_size * (block_size + 1) * linear_block_count >= target_coverage:
        linear_block_count += 1
        block_size = depth // linear_block_count
    linear_block_count -= 1
    if not linear_block_count:
        linear_block_count = 1

    full_depth, residual_depth = depth // linear_block_count, depth % linear_block_count
    blocks = (full_depth,) * linear_block_count + ((residual_depth,) if residual_depth else ())
    blocks = tuple(1 + i for depth in blocks for i in range(depth))

    main = [reversible_module(hidden_features,
                              hidden_features,
                              b * (not transpose) + 1,
                              transpose,
                              kernel_size,
                              heads,
                              revnet=revnet,
                              top_norm=bool(i) or in_features == hidden_features)
            for b,i in enumerate(blocks)]

    if in_features != hidden_features:
        main.insert(0, PositionalSeparableConvolution(in_features, hidden_features))
    if out_features is not None and hidden_features != out_features:
        main.append(PositionalSeparableConvolution(hidden_features, out_features))
    if end is not None:
        main.append(end)
    if wrap:
        main = torch.nn.Sequential(*main)
    return main
