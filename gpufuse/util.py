"""helper functions
"""
import os

import tifffile as tf
import numpy as np
from scipy.interpolate import interp1d

from .func import decon_dualview, reg_3dgpu


def fuse(im_a, im_b, psf_a, psf_b, **kwargs):
    h_reg_b = reg_3dgpu(im_a, im_b, **kwargs)[0]
    decon_result = decon_dualview(im_a, h_reg_b, psf_a, psf_b, **kwargs)[0]
    return decon_result, h_reg_b


def fuse_file(imname, psfname):
    im_a = tf.imread(imname)
    im_b = tf.imread(imname.replace("A_", "B_"))
    psf_a = tf.imread(psfname)
    psf_b = tf.imread(psfname.replace("A_", "B_"))
    psf_a_bp = None
    if os.path.exists(psfname.replace(".tif", "_BP.tif")):
        psf_a_bp = tf.imread(psfname.replace(".tif", "_BP.tif"))
    psf_b_bp = None
    if os.path.exists(psfname.replace("A_", "B_").replace(".tif", "_BP.tif")):
        psf_b_bp = tf.imread(psfname.replace("A_", "B_").replace(".tif", "_BP.tif"))
    decon, reg = fuse(im_a, im_b, psf_a, psf_b, psf_a_bp=psf_a_bp, psf_b_bp=psf_b_bp)
    return decon, reg


def resize_axis(im, newZ, axis=0, kind="cubic"):
    nZ = im.shape[axis]
    x = np.arange(0, nZ)
    f = interp1d(x, im, axis=axis, kind=kind)
    xx = np.linspace(0, nZ - 1, newZ)
    return f(xx)