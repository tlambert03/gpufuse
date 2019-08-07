"""python functions wrapping c++ interfaces in lib.py
"""
import ctypes
import warnings

import numpy as np

from .lib import LIB


def query_device():
    """Print GPU info to console."""
    LIB.queryDevice()


def get_tif_info(fname):
    """Get TIFF file information.

    Args:
        fname (str): tiff file path

    Returns:
        (tuple): 2-tuple containing:

            int: bit depth of image
            list: [X, Y, Z] dimensions of tiff
    """
    size = (ctypes.c_uint * 3)()
    bit_per_sample = LIB.gettifinfo(fname.encode(), size)
    return bit_per_sample, list(size)


def read_tif_stack(fname):
    """Read TIFF file, only 16-bit and 32-bit are supported, image size < 4 GB.
    Note: this function is much slower than just using tifffile.imread

    Args:
        fname (str): tiff file path

    Returns:
        np.ndarray: tiff stack
    """
    size = (ctypes.c_uint * 3)()
    bps, shape = get_tif_info(fname)
    result = np.empty(tuple(reversed(shape)), dtype=np.float32)
    LIB.readtifstack(result, fname.encode(), size)
    return result.astype(np.dtype("uint%d" % bps))


def reg_3dgpu(
    im_a,
    im_b,
    tmx=None,
    reg_method=7,
    i_tmx=0,
    FTOL=0.0001,
    itLimit=3000,
    sub_background=True,
    device_num=0,
    nptrans=False,
    **kwargs  # just here to catch extra keyward arguments from util.fuse
):
    """Register two 3D numpy arrays on GPU.

    Args:
        im_a (np.ndarray): target image
        im_b (np.ndarray): source image, will be registered to target
        tmx (np.ndarray, optional): Transformation matrix. Defaults to np.eye(4).
        reg_method (int, optional): Registration Method. Defaults to 7. one of:
            0 - no registration, transform image 2 based on input matrix
            1 - translation only
            2 - rigid body
            3 - 7 degrees of freedom (translation, rotation,
                scaling equally in 3 dimensions)
            4 - 9 degrees of freedom (translation, rotation, scaling)
            5 - 12 degrees of freedom
            6 - rigid body first, then do 12 degrees of freedom
            7 - 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
        i_tmx (int, optional): input transformation matrix. Defaults to 0.
            0 - use default initial matrix (h_img1 and h_img2 are aligned at center)
            1 - use iTmx as initial matrix
            2 - do translation registration based on phase information
            3 - do translation registration based on 2D max projections
                (there is a bug with this option).
        FTOL (float, optional): convergence threshold, defined as the difference
                                of the cost function. value from two adjacent iterations.
                                Defaults to 0.0001.
        itLimit (int, optional): maximum iteration number. Defaults to 3000.
        sub_background (bool, optional): subject background before registration or not.
                                         Defaults to True.
        device_num (int, optional): the GPU device to be used, 0-based naming convention.
                                    Defaults to 0.
        nptrans (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: If reg_method < 0 or > 7
        ValueError: If i_tmx < 0 or > 3

    Returns:
        tuple: 3-tuple containing:

            np.ndarray: the registered volume
            list: the transformation matrix
            list: 11-element array, records and feedbacks of the processing
                  [0]-[3]:  initial GPU memory, after variables allocated, after
                            processing, after variables released (all in MB)
                  [4]-[6]:  initial cost function value, minimized cost function value,
                            intermediate cost function value
                  [7]-[10]: registration time (s), whole time (s), single sub iteration
                            time (ms), total sub iterations
    """
    if not 0 <= reg_method <= 7:
        raise ValueError("reg_method must be between 0-7")
    if not 0 <= i_tmx <= 3:
        raise ValueError("i_tmx must be between 0-3")
    if nptrans:
        im_a = np.ascontiguousarray(np.transpose(im_a, (2, 1, 0)), dtype=np.float32)
        im_b = np.ascontiguousarray(np.transpose(im_b, (2, 1, 0)), dtype=np.float32)
    h_reg_b = np.empty_like(im_a, dtype=np.float32)
    if tmx is None:
        tmx = np.eye(4)
    tmx = (ctypes.c_float * 16)(*tmx.ravel())

    h_records = (ctypes.c_float * 11)(0.0)
    status = LIB.reg_3dgpu(
        h_reg_b,
        tmx,
        im_a.astype(np.float32),
        im_b.astype(np.float32),
        # reversed shape because numpy is ZYX, and c++ expects XYZ
        (ctypes.c_uint * 3)(*reversed(im_a.shape)),
        (ctypes.c_uint * 3)(*reversed(im_a.shape)),
        reg_method,
        i_tmx,
        FTOL,
        itLimit,
        int(sub_background),
        device_num,
        h_records,
    )
    if status > 0:
        warnings.warn("CUDA status not 0")
    if nptrans:
        h_reg_b = np.transpose(h_reg_b, (2, 1, 0))
    return h_reg_b, list(tmx), list(h_records)


def decon_dualview(
    im_a,
    im_b,
    psf_a,
    psf_b,
    psf_a_bp=None,
    psf_b_bp=None,
    iters=10,
    device_num=0,
    gpu_mem_mode=0,
):
    """3D joint Richardson-Lucy deconvolution with GPU implementation,
    compatible with unmatched back projector.

    the two view images (``im_a`` and ``im_b``) are assumed to have isotripic pixel size
    and oriented in the same direction.

    Args:
        im_a (np.ndarray): input image 1
        im_b (np.ndarray): input image 2
        psf_a (np.ndarray): forward projector 1 (PSF 1)
        psf_b (np.ndarray): forward projector 2 (PSF 2)
        psf_a_bp (np.ndarray, optional): unmatched back projector corresponding to psf_a
                                         Defaults to None.
        psf_b_bp (np.ndarray, optional): unmatched back projector corresponding to psf_b
                                         Defaults to None.
        iters (int, optional): number of iterations in deconvolution. Defaults to 10.
        device_num (int, optional): the GPU device to be used, 0-based naming convention.
                                    Defaults to 0.
        gpu_mem_mode (int, optional): the GPU memory mode. Defaults to 0.
            0 - Automatically set memory mode based on calculations
            1 - sufficient memory
            2 - memory optimized

    Returns:
        tuple: 2-tuple containing:

            np.ndarray: the fused & deconvolved result
            list: 10-element array, records and feedbacks of the processing
                [0]      the actual memory mode used;
                [1]-[5]  initial GPU memory, after variables partially allocated,
                         during processing, after processing, after variables released
                         (all in MB)
                [6]-[9]  initializing time, prepocessing time, decon time, total time
    """
    h_decon = np.empty_like(im_a, dtype=np.float32)
    decon_records = (ctypes.c_float * 10)(0.0)
    if isinstance(psf_a_bp, np.ndarray) and isinstance(psf_b_bp, np.ndarray):
        unmatched = True
    else:
        unmatched = False
        psf_a_bp = psf_a
        psf_b_bp = psf_b
    status = LIB.decon_dualview(
        h_decon,
        im_a.astype(np.float32),
        im_b.astype(np.float32),
        # reversed shape because numpy is ZYX, and c++ expects XYZ
        (ctypes.c_uint * 3)(*reversed(im_a.shape)),
        psf_a.astype(np.float32),
        psf_b.astype(np.float32),
        (ctypes.c_uint * 3)(*reversed(psf_a.shape)),
        iters,
        device_num,
        gpu_mem_mode,
        decon_records,
        unmatched,
        psf_a_bp,
        psf_b_bp,
    )
    if status > 0:
        warnings.warn("CUDA status not 0")
    return h_decon, list(decon_records)
