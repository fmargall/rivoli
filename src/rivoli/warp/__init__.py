# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Parallel kernels written with NVIDIA Warp.

The modules of this package compute the same quantities as their NumPy
counterparts in :mod:`rivoli`, in parallel on a CUDA GPU or on the CPU.
They require the ``warp`` extra: ``pip install rivoli[warp]``.

Their device is chosen by the ``device`` argument of the functions, or
else by the default device of Warp, which is the first CUDA GPU when
there is one.
"""
