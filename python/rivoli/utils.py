import numpy as np

from . import _binding

def rusinkiewiczToSpherical(thetaHArray: float,
                            thetaDArray: float,
                            phiDArray  : float, **kwargs):
    dtype = kwargs.get("dtype", "float32")

    if   dtype in (np.float32, "float32", "f32"):
        dtypeStr = "f32"
    elif dtype in (np.float64, "float64", "f64"):
        dtypeStr = "f64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    fctName = f"_rusinkiewicz_to_spherical_None_{dtypeStr}"

    try:
        fct = getattr(_binding.utils, fctName)
    except AttributeError:
        raise ValueError(f"Unsupported configuration: {fctName}")

    return fct(thetaHArray, thetaDArray, phiDArray)

def sphericalToRusinkiewicz(thetaIArray  : float,
                            thetaOArray  : float,
                            deltaPhiArray: float, **kwargs):
    dtype = kwargs.get("dtype", "float32")

    if   dtype in (np.float32, "float32", "f32"):
        dtypeStr = "f32"
    elif dtype in (np.float64, "float64", "f64"):
        dtypeStr = "f64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    fctName = f"_spherical_to_rusinkiewicz_None_{dtypeStr}"

    try:
        fct = getattr(_binding.utils, fctName)
    except AttributeError:
        raise ValueError(f"Unsupported configuration: {fctName}")

    return fct(thetaIArray, thetaOArray, deltaPhiArray)
