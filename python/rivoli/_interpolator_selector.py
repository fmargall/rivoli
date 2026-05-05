import numpy as np

from . import _binding

def _build_interpolator(coordinates, values, **kwargs):
    kernel        = kwargs.get("kernel", "linear")
    bilateral     = kwargs.get("bilateral", False)
    reciprocal    = kwargs.get("reciprocal", False)
    nonNegativity = kwargs.get("nonNegativity", False)
    isotropy      = kwargs.get("isotropy", False)
    backend       = kwargs.get("backend", "None")
    dtype         = kwargs.get("dtype", "float32")
    tikhonov      = kwargs.get("tikhonov", 0.0)
    nbSamples     = kwargs.get("nbSamples", 0)

    parameterisation = kwargs.get("parameterisation", "spherical")

    dimension = coordinates.shape[1]

    # -------- Kernel selection --------
    kernelSpecs = {
        "linear":       {"name": "Linear",       "params": []},
        "cubic":        {"name": "Cubic",        "params": []},
        "epanechnikov": {"name": "Epanechnikov", "params": ["sigma"]},
        "gaussian":     {"name": "Gaussian",     "params": ["sigma"]},
        "laplacian":    {"name": "Laplacian",    "params": ["sigma"]},

        # Anisotropic kernels
        "anisotropicgaussian":  {"name": "AnisotropicGaussian",  "params": ["sigmaOne", "sigmaTwo"]},
        "anisotropiclaplacian": {"name": "AnisotropicLaplacian", "params": ["sigmaOne", "sigmaTwo"]},
    }
    try:
        spec = kernelSpecs[kernel.lower()]
    except KeyError:
        raise ValueError(f"Unsupported kernel: {kernel}")
    kernelStr = spec["name"]

    missingParams = [p for p in spec["params"] if p not in kwargs]
    if missingParams:
        raise ValueError(f"{kernel} kernel requires parameters: {missingParams}")
    kernelParams = [kwargs[p] for p in spec["params"]]

    # -------- Topology selection --------
    if dimension == 2:
        topologyStr = "2S"
        if bilateral:
            topologyStr += "Bilateral"

    elif dimension == 3:
        topologyStr = "1Rx2S"
    elif dimension == 4:
        topologyStr = "2Sx2S"

    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    if dimension == 3 or dimension == 4:
        if bilateral:
            topologyStr += "Bilateral"
        if reciprocal:
            topologyStr += "Reciprocal"

        if parameterisation == "rusinkiewicz":
            topologyStr += "Rusinkiewicz"

        if isotropy:
            topologyStr += "Isotropic"

        # Currently only Euclidean distance is supported
        topologyStr += "Euclidean"

    # -------- Backend selection --------
    backendStr = {
        "None" : "None" ,
        "SSE41": "SSE41",
        "AVX"  : "AVX"
    }[backend]

    # -------- Dtype selection --------
    if   dtype in (np.float32, "float32", "f32"):
        dtypeStr = "f32"
    elif dtype in (np.float64, "float64", "f64"):
        dtypeStr = "f64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    className = f"_Interpolator_{topologyStr}_{kernelStr}_{backendStr}_{dtypeStr}"

    try:
        cls = getattr(_binding, className)
    except AttributeError:
        raise ValueError(f"Unsupported configuration: {className}")

    return cls(coordinates, values, tikhonov, nonNegativity, nbSamples, *kernelParams)
