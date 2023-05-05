import math


def get_conv_out_size(h, w, kernel_size, stride, dilation=1, padding=0):
    """Note that in PyTorch the image is channel x height x width"""

    if type(kernel_size) == tuple:
        kernel_size_h, kernel_size_w = kernel_size
    elif type(kernel_size) == int:
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size
    else:
        raise AssertionError("Kernel size must either be tuple with 2 values or int")

    if type(padding) == tuple:
        padding_h, padding_w = padding
    elif type(padding) == int:
        padding_h = padding
        padding_w = padding
    else:
        raise AssertionError("Padding must either be tuple with 2 values or int")

    if type(dilation) == tuple:
        dilation_h, dilation_w = dilation
    elif type(dilation) == int:
        dilation_h = dilation
        dilation_w = dilation
    else:
        raise AssertionError("Dilation must either be tuple with 2 values or int")

    if type(stride) == tuple:
        stride_h, stride_w = stride
    elif type(stride) == int:
        stride_h = stride
        stride_w = stride
    else:
        raise AssertionError("Stride must either be tuple with 2 values or int")

    h_out = int(
        math.floor(
            (h + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / float(stride_h)
            + 1
        )
    )

    w_out = int(
        math.floor(
            (w + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / float(stride_w)
            + 1
        )
    )

    return h_out, w_out
