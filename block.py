import torch
from memcnn import AdditiveCoupling, InvertibleModuleWrapper
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, ConvTranspose2d, Linear, Module


@torch.jit.script
def mish(fn_input: torch.Tensor) -> torch.Tensor:
    """
    Mish activation function: https://arxiv.org/abs/1908.08681
    :param fn_input: tensor about to be activated
    :return: activated tensor
    """
    return torch.nn.functional.softplus(fn_input).tanh().mul(fn_input)


activation = mish


def arange_like(dim: int, tensor: torch.Tensor) -> torch.Tensor:
    """
    Creates a pytorch range with shape similar to that of the input tensor, where the indexes go up along dim
    :param dim: Dimension to index
    :param tensor: Input tensor
    :return: Index tensor
    """
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
    """
    Helper class to normalize and activate inputs
    """

    def __init__(self, features: int):
        super(Activate, self).__init__()
        self.norm = BatchNorm2d(features)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        """
        Function overwriting pytorch's forward method. Activates a tensor.
        :param fn_input: Tensor about to be activated
        :return: Activated tensor
        """
        return activation(self.norm(fn_input))


class SeparableConvolution(Module):
    """
    Depthwise separable convolution (or XCeption): https://arxiv.org/pdf/1610.02357.pdf
    """

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
        """
        Function overwriting pytorch's forward method. Computes convolutions on input.
        :param fn_input: Any tensor
        :return: Output from convolution
        """
        return self.point(activation(self.mid_norm(self.depth(fn_input))))


class PositionalSeparableConvolution(SeparableConvolution):
    """
    XCeption with positional information
    """

    def __init__(self, in_features: int, out_features: int, stride=1, dilation=1,
                 transpose=False, kernel_size=3):
        super(PositionalSeparableConvolution, self).__init__(4 + in_features,
                                                             out_features,
                                                             stride, dilation,
                                                             transpose, kernel_size)

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        """
        Function overwriting pytorch's forward method. Computes convolutions on input.
        :param fn_input: Any tensor
        :return: Output from convolution
        """
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
    """
    Doubled convolution with normalization between, optional input normalization and
    optional multihead squeeze-excitation (similar to attention).
    Doubled convolution is required as it's wrapped by residual or reversible blocks.
    """

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
        """
        Function overwriting pytorch's forward method. Computes convolutions on input.
        :param fn_input: Any tensor
        :return: Output from convolution
        """
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


class RandomPad(Module):
    """
    Pad _any_ dimension by a constant amount.
    """

    def __init__(self, dim, amount):
        super(RandomPad, self).__init__()
        self.dim = dim
        self.amount = amount

    def forward(self, fn_input: torch.Tensor) -> torch.Tensor:
        """
        Function overwriting pytorch's forward method. Pads input tensor at given dimension with pad value.
        :param fn_input: Any tensor
        :param pad: Constant value to pad with
        :return: Padded tensor
        """
        size = list(fn_input.size())
        size[self.dim] = self.amount
        pad = torch.randn(size, device=fn_input.device, requires_grad=False)
        out = torch.cat([fn_input, pad], self.dim)
        return out

    def inverse(self, fn_input: torch.Tensor) -> torch.Tensor:
        """
        Inverse padding (removal of previously added padding)
        :param fn_input: Any tensor
        :return: Un-padded tensor
        """
        return torch.cat([fn_input.select(self.dim, i) for i in range(fn_input.size(self.dim) - self.amount)], self.dim)

    def __str__(self):
        return f'{type(self).__name__}({self.dim}, {self.amount})'

    def __repr__(self):
        return str(self)


def reversible_module(inf, outf, *args, stride=1, revnet=False, **kwargs) -> Module:
    """
    Create a reversible BasicModule with given parameters
    :param inf: Input features
    :param outf: Output features
    :param args: Other positional arguments passed to BasicModule
    :param stride: Stride
    :param revnet: Whether to try using revnet architecture
    :param kwargs: Keyword arguments passed to BasicModule
    :return: Created Module
    """
    if revnet and stride == 1:
        inf //= 2
        outf //= 2
    module = BasicModule(inf, outf, *args, **kwargs, stride=stride)
    if revnet and stride == 1:
        return InvertibleModuleWrapper(AdditiveCoupling(module))
    return module


def linear_dilated_model(features, depth=4, transpose=False, kernel_size=3, heads=12, target_coverage=1):
    """
    Create a full model with linearly increasing dilation
    :param features: Number of features used throughout the model
    :param depth: Number of blocks (not layers!) in the model
    :param transpose: Whether all convolutions are transposed or not
    :param kernel_size: Size of convolution kernel
    :param heads: Number of heads used in BasicModule attention (0 = off)
    :param target_coverage: Target coverage area, used to calculate dilation
    :return: List of reversible blocks
    """
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

    main = [reversible_module(features,
                              features,
                              (b + 1) * (not transpose) + 1,
                              transpose,
                              kernel_size,
                              heads,
                              revnet=True,
                              top_norm=bool(i))
            for b, i in enumerate(i for depth in blocks for i in range(depth))]
    return main
