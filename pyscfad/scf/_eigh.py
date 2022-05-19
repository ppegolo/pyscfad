from functools import partial
import numpy
import scipy.linalg
from jax import custom_jvp
from jax import jit
from pyscfad import lib
from pyscfad.lib import ops
from pyscfad.lib import numpy as np

DEG_THRESH = 1e-10

def eigh(a, b=None, x0=None, eigvals_only=False, **kwargs):
    if x0 is None:
        return lib.linalg_helper.eigh(a, b, eigvals_only=eigvals_only, **kwargs)

    a = 0.5 * (a + a.T.conj())
    if b is not None:
        b = 0.5 * (b + b.T.conj())

    w, v =  _eigh(a, b, x0)

    if eigvals_only:
        return w
    else:
        return w, v

@partial(custom_jvp, nondiff_argnums=(2,))
def _eigh(a, b, x0):
    w, v = scipy.linalg.eigh(a, b=b)
    return w, v

@_eigh.defjvp
def _eigh_jvp(v, primals, tangents):
    a, b = primals
    at, bt = tangents
    w, _ = primal_out = _eigh(a, b, v)

    deg_thresh = DEG_THRESH
    eji = w[..., numpy.newaxis, :] - w[..., numpy.newaxis]
    idx = abs(eji) < deg_thresh
    #eji[idx] = 1.e200
    #eji[numpy.diag_indices_from(eji)] = 1
    eji = ops.index_update(eji, idx, 1.e200)
    eji = ops.index_update(eji, np.diag_indices_from(eji), 1.)
    eye_n = numpy.eye(a.shape[-1], dtype=a.dtype)
    Fmat = np.reciprocal(eji) - eye_n
    if b is None:
        dw, dv = _eigh_jvp_jitted_nob(v, Fmat, at)
    else:
        dw, dv = _eigh_jvp_jitted(w, v, Fmat, at, bt)
    return primal_out, (dw,dv)

@jit
def _eigh_jvp_jitted(w, v, Fmat, at, bt):
    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    vt_bt_v = np.dot(v.conj().T, np.dot(bt, v))
    vt_bt_v_w = np.dot(vt_bt_v, np.diag(w))
    da_minus_ds = vt_at_v - vt_bt_v_w
    dw = np.diag(da_minus_ds).real

    eye_n = np.eye(vt_bt_v.shape[-1])
    dv = np.dot(v, np.multiply(Fmat, da_minus_ds) - np.multiply(eye_n, vt_bt_v) * .5)
    return dw, dv

@jit
def _eigh_jvp_jitted_nob(v, Fmat, at):
    vt_at_v = np.dot(v.conj().T, np.dot(at, v))
    dw = np.diag(vt_at_v)
    dv = np.dot(v, np.multiply(Fmat, vt_at_v))
    return dw, dv