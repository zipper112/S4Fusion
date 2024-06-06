from torch.nn import Module
from torch.nn import functional as F

from torch import Tensor
from torch.nn.common_types import _size_any_t

__all__ = ['Fold', 'Unfold']

class Fold(Module):
  
    __constants__ = ['output_size', 'kernel_size', 'dilation', 'padding',
                     'stride']
    output_size: _size_any_t
    kernel_size: _size_any_t
    dilation: _size_any_t
    padding: _size_any_t
    stride: _size_any_t

    def __init__(
        self,
        kernel_size: _size_any_t,
        dilation: _size_any_t = 1,
        padding: _size_any_t = 0,
        stride: _size_any_t = 1
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor, output_size) -> Tensor:
        return F.fold(input, output_size, self.kernel_size, self.dilation,
                      self.padding, self.stride)

    def extra_repr(self) -> str:
        return 'output_size={output_size}, kernel_size={kernel_size}, ' \
            'dilation={dilation}, padding={padding}, stride={stride}'.format(
                **self.__dict__
            )