from functools import partial
import numpy as onp
#from jax import numpy
#from jax.config import config as jax_config
from pyscf import numpy
from pyscf.lib import ops
from .jax_helper import jit, vmap
#jax_config.update("jax_enable_x64", True)

einsum = numpy.einsum
dot = numpy.dot

__all__ = ['numpy', 'einsum', 'dot',
           'PLAIN', 'HERMITIAN', 'ANTIHERMI', 'SYMMETRIC',
           'unpack_triu', 'unpack_tril', 'pack_tril']

PLAIN = 0
HERMITIAN = 1
ANTIHERMI = 2
SYMMETRIC = 3

@partial(jit, static_argnums=1)
def _unpack_triu(triu, filltril=HERMITIAN):
    '''
    Unpack the upper triangular part of a matrix
    '''
    assert triu.ndim == 1
    nd = int(onp.sqrt(2*triu.size))
    out = numpy.zeros((nd,nd), dtype=triu.dtype)
    idx = onp.triu_indices(nd)
    out = ops.index_update(out, idx, triu)
    if filltril == PLAIN:
        return out
    elif filltril == HERMITIAN:
        out += numpy.tril(out.T.conj(), -1)
        return out
    elif filltril == ANTIHERMI:
        out -= out.conj().T
        return out
    elif filltril == SYMMETRIC:
        out += numpy.tril(out.T, -1)
        return out
    else:
        raise KeyError

def unpack_triu(triu, filltril=HERMITIAN, axis=-1, out=None):
    if triu.ndim == 1:
        out = _unpack_triu(triu, filltril)
    elif triu.ndim == 2:
        if axis == -1 or axis == 1:
            out = vmap(_unpack_triu, (0,None))(triu, filltril)
        elif axis == 0 or axis == -2:
            out = vmap(_unpack_triu, (1,None))(triu, filltril)
    else:
        raise NotImplementedError
    return out

@partial(jit, static_argnums=1)
def _unpack_tril(tril, filltriu=HERMITIAN):
    '''
    Unpack the lower triangular part of a matrix
    '''
    assert tril.ndim == 1
    nd = int(onp.sqrt(2*tril.size))
    out = numpy.zeros((nd,nd), dtype=tril.dtype)
    idx = onp.tril_indices(nd)
    out = ops.index_update(out, idx, tril)
    if filltriu == PLAIN:
        return out
    elif filltriu == HERMITIAN:
        out += numpy.triu(out.T.conj(), 1)
        return out
    elif filltriu == ANTIHERMI:
        out -= out.T.conj()
        return out
    elif filltriu == SYMMETRIC:
        out += numpy.triu(out.T, 1)
        return out
    else:
        raise KeyError

def unpack_tril(tril, filltriu=HERMITIAN, axis=-1, out=None):
    if tril.ndim == 1:
        out = _unpack_tril(tril, filltriu)
    elif tril.ndim == 2:
        if axis == -1 or axis == 1:
            out = vmap(_unpack_tril, (0,None), signature='(n)->(m,m)')(tril, filltriu)
        elif axis == 0 or axis == -2:
            out = vmap(_unpack_tril, (1,None), signature='(n)->(m,m)')(tril, filltriu)
    else:
        raise NotImplementedError
    return out

@partial(jit, static_argnums=1)
def pack_tril(a, axis=-1):
    '''
    Lower triangular part of a matrix as a vector
    '''
    def fn(mat):
        idx = onp.tril_indices(mat.shape[0])
        return mat[idx].ravel()

    if a.ndim == 3:
        if axis == -1:
            tril = vmap(fn, signature='(m,m)->(n)')(a)
        elif axis == 0:
            tril = vmap(fn, -1, signature='(m,m)->(n)')(a)
        else:
            raise KeyError
    else:
        raise NotImplementedError
    return tril
