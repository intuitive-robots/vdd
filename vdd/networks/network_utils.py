from typing import Iterable, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer

import numpy as np

def str2torchdtype(str_dtype: str = 'float32'):
    if str_dtype == 'float32':
        return torch.float32
    elif str_dtype == 'float64':
        return torch.float64
    elif str_dtype == 'float16':
        return torch.float16
    elif str_dtype == 'int32':
        return torch.int32
    elif str_dtype == 'int64':
        return torch.int64
    else:
        raise NotImplementedError


def inverse_softplus(x):

    """
    x = inverse_softplus(softplus(x))
    Args:
        x: data

    Returns:

    """
    return (x.exp() - 1.).log()

def fill_triangular(x, upper=False):
    """
    From: https://github.com/tensorflow/probability/blob/c833ee5cd9f60f3257366b25447b9e50210b0590/tensorflow_probability/python/math/linalg.py#L787
    License: Apache-2.0

    Creates a (batch of) triangular matrix from a vector of inputs.

    Created matrix can be lower- or upper-triangular. (It is more efficient to
    create the matrix as upper or lower, rather than transpose.)

    Triangular matrix elements are filled in a clockwise spiral. See example,
    below.

    If `x.shape` is `[b1, b2, ..., bB, d]` then the output shape is
    `[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
    `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

    Example:

    ```python
    fill_triangular([1, 2, 3, 4, 5, 6])
    # ==> [[4, 0, 0],
    #      [6, 5, 0],
    #      [3, 2, 1]]

    fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
    # ==> [[1, 2, 3],
    #      [0, 5, 6],
    #      [0, 0, 4]]
    ```

    The key trick is to create an upper triangular matrix by concatenating `x`
    and a tail of itself, then reshaping.

    Suppose that we are filling the upper triangle of an `n`-by-`n` matrix `M`
    from a vector `x`. The matrix `M` contains n**2 entries total. The vector `x`
    contains `n * (n+1) / 2` entries. For concreteness, we'll consider `n = 5`
    (so `x` has `15` entries and `M` has `25`). We'll concatenate `x` and `x` with
    the first (`n = 5`) elements removed and reversed:

    ```python
    x = np.arange(15) + 1
    xc = np.concatenate([x, x[5:][::-1]])
    # ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13,
    #            12, 11, 10, 9, 8, 7, 6])

    # (We add one to the arange result to disambiguate the zeros below the
    # diagonal of our upper-triangular matrix from the first entry in `x`.)

    # Now, when reshapedlay this out as a matrix:
    y = np.reshape(xc, [5, 5])
    # ==> array([[ 1,  2,  3,  4,  5],
    #            [ 6,  7,  8,  9, 10],
    #            [11, 12, 13, 14, 15],
    #            [15, 14, 13, 12, 11],
    #            [10,  9,  8,  7,  6]])

    # Finally, zero the elements below the diagonal:
    y = np.triu(y, k=0)
    # ==> array([[ 1,  2,  3,  4,  5],
    #            [ 0,  7,  8,  9, 10],
    #            [ 0,  0, 13, 14, 15],
    #            [ 0,  0,  0, 12, 11],
    #            [ 0,  0,  0,  0,  6]])
    ```

    From this example we see tht the resuting matrix is upper-triangular, and
    contains all the entries of ax, as desired. The rest is details:

    - If `n` is even, `x` doesn't exactly fill an even number of rows (it fills
      `n / 2` rows and half of an additional row), but the whole scheme still
      works.
    - If we want a lower triangular matrix instead of an upper triangular,
      we remove the first `n` elements from `x` rather than from the reversed
      `x`.

    For additional comparisons, a pure numpy version of this function can be found
    in `distribution_util_test.py`, function `_fill_triangular`.

    Args:
      x: `Tensor` representing lower (or upper) triangular elements.
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
      tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

    Raises:
      ValueError: if `x` cannot be mapped to a triangular matrix.
    """

    m = np.int32(x.shape[-1])
    # Formula derived by solving for n: m = n(n+1)/2.
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
        raise ValueError('Input right-most shape ({}) does not '
                         'correspond to a triangular matrix.'.format(m))
    n = np.int32(n)
    new_shape = x.shape[:-1] + (n, n)

    ndims = len(x.shape)
    if upper:
        x_list = [x, torch.flip(x[..., n:], dims=[ndims - 1])]
    else:
        x_list = [x[..., n:], torch.flip(x, dims=[ndims - 1])]

    x = torch.cat(x_list, dim=-1).reshape(new_shape)
    x = torch.triu(x) if upper else torch.tril(x)
    return x

def diag_bijector(f: callable, x):
    """
    Apply transformation f(x) on the diagonal of a batched matrix.
    Args:
        f: callable to apply to diagonal
        x: data

    Returns:
        transformed matrix x
    """
    return x.tril(-1) + f(x.diagonal(dim1=-2, dim2=-1)).diag_embed() + x.triu(1)

def get_optimizer(optimizer_type: str, model_parameters: Union[Iterable[torch.Tensor], Iterable[dict]],
                  learning_rate: float, **kwargs):
    """
    Get optimizer instance for given model parameters
    Args:
        model_parameters:
        optimizer_type:
        learning_rate:
        **kwargs:

    Returns:

    """
    if optimizer_type.lower() == "sgd":
        return optim.SGD(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "sgd_momentum":
        momentum = kwargs.pop("momentum") if kwargs.get("momentum") else 0.9
        return optim.SGD(model_parameters, learning_rate, momentum=momentum, **kwargs)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(model_parameters, learning_rate, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(model_parameters, learning_rate, betas=(0.95, 0.999), eps=1e-8, **kwargs)
    elif optimizer_type.lower() == "adagrad":
        return optim.adagrad.Adagrad(model_parameters, learning_rate, **kwargs)
    else:
        ValueError(f"Optimizer {optimizer_type} is not supported.")


def get_lr_schedule(schedule_type: str, optimizer: Optimizer, total_iters) -> Union[
    optim.lr_scheduler._LRScheduler, None]:
    if not schedule_type or schedule_type.isspace():
        return None

    elif schedule_type.lower() == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0., total_iters=total_iters)

    elif schedule_type.lower() == "papi":
        # Multiply learning rate with 0.8 every time the backtracking fails
        return optim.lr_scheduler.MultiplicativeLR(optimizer, lambda n_calls: 0.8)

    elif schedule_type.lower() == "performance":
        return optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.8), \
               optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 1.01)

    else:
        raise ValueError(
            f"Learning rate schedule {schedule_type} is not supported. Select one of [None, linear, papi, performance].")