import os

import numpy as np
import tifffile as tf

import gpufuse


def test_reg():
    basedir = os.path.join(os.path.dirname(__file__), "sample_data")
    im_a = tf.imread(os.path.join(basedir, "StackA_0.tif"))
    im_b = tf.imread(os.path.join(basedir, "StackB_0.tif"))
    result, tmx, record = gpufuse.reg_3dgpu(im_a, im_b)
    assert result.shape == (246, 360, 240)
    assert len(tmx) == 16
    assert len(record) == 11


def test_fuse():
    basedir = os.path.join(os.path.dirname(__file__), "sample_data")
    im_a = tf.imread(os.path.join(basedir, "StackA_0.tif"))
    im_b = tf.imread(os.path.join(basedir, "StackB_0.tif"))
    psf_a = tf.imread(os.path.join(basedir, "PSFA.tif"))
    psf_b = np.ascontiguousarray(np.transpose(psf_a, (2, 1, 0)))
    result, reg = gpufuse.fuse(im_a, im_b, psf_a, psf_b)
    assert result.shape == (246, 360, 240)
    assert reg.shape == (246, 360, 240)
