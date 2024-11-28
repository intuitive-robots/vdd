import torch
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