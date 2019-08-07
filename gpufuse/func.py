"""python functions wrapping c++ interfaces in lib.py
"""
import ctypes
import warnings
import os
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
    ftol=0.0001,
    reg_iters=3000,
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
        ftol (float, optional): convergence threshold, defined as the difference
                                of the cost function. value from two adjacent iterations.
                                Defaults to 0.0001.
        reg_iters (int, optional): maximum iteration number. Defaults to 3000.
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
        ftol,
        reg_iters,
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


def fetch_pixelsize(folder):
    out = (ctypes.c_float * 3)(0.1625, 0.1625, 1.0)
    return out


def fusion_dualview_batch(
    spim_a_folder,
    psf_a,
    psf_b=None,
    spim_b_folder=None,
    psf_a_bp=None,
    psf_b_bp=None,
    out_path=None,
    name_a="SPIMA_",
    name_b="SPIMB_",
    nstart=0,
    nend=None,
    ntest=0,
    interval=1,
    reg_mode=3,
    rot_mode=0,
    tmx_mode=0,
    itmx=None,
    ftol=0.0001,
    reg_iters=3000,
    iters=10,  # for deconvolution
    device_num=0,
    save_inter=False,
    save_reg=[False, False],  # path A, path B
    save_mips=[True, False, False],  # [Z, Y ,X]
    save_3d_mips=[False, False],  # decon 3D mip [Y, X]
    output_bit=16,
):
    if not 0 <= reg_mode <= 3:
        raise ValueError("reg_mode must be between 0-3")
    if not -1 <= rot_mode <= 1:
        raise ValueError("rot_mode must be between -1 and 1")
    if not 0 <= tmx_mode <= 3:
        raise ValueError("tmx_mode must be between 0-3")
    if output_bit not in (16, 32):
        raise ValueError("output_bit must be either 16 or 32")

    if spim_b_folder is None:
        spim_b_folder = spim_a_folder.replace("SPIMA", "SPIMB")
    if not os.path.isdir(spim_b_folder):
        raise FileNotFoundError("spim_b_folder not found")

    if psf_b is None:
        psf_b = psf_a.replace("A", "B").replace("a", "b")
    if not os.path.exists(psf_b):
        raise FileNotFoundError("psf_b not found")

    if not spim_a_folder.endswith(os.sep):
        spim_a_folder += os.sep
    if not spim_b_folder.endswith(os.sep):
        spim_b_folder += os.sep

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(spim_a_folder)), "result_"
        )

    tot_n = len(
        [
            x
            for x in os.listdir(spim_a_folder)
            if (x.endswith(".tif") or x.endswith(".tiff"))
        ]
    )
    if nend is None:
        nend = tot_n - 1
    if nend > (tot_n - 1):
        raise ValueError(
            "nend is greater than the number of images in {}".format(spim_a_folder)
        )

    pixel_size1 = fetch_pixelsize(spim_a_folder)
    pixel_size2 = fetch_pixelsize(spim_b_folder)

    if tmx_mode != 1 or itmx is None:
        itmx = np.eye(4)
    itmx = (ctypes.c_float * 16)(*itmx.ravel())

    saves = (ctypes.c_uint * 8)(save_inter, *save_reg, *save_mips, *save_3d_mips)
    records = (ctypes.c_float * 22)(0)

    if psf_a_bp and psf_b_bp and os.path.exists(psf_a_bp) and os.path.exists(psf_b_bp):
        unmatched = True
    else:
        unmatched = False
        psf_a_bp = psf_a
        psf_b_bp = psf_b

    LIB.fusion_dualview_batch(
        out_path.encode(),
        spim_a_folder.encode(),
        spim_b_folder.encode(),
        name_a.encode(),
        name_b.encode(),
        nstart,
        nend,
        interval,
        ntest,
        pixel_size1,
        pixel_size2,
        reg_mode,
        rot_mode,
        tmx_mode,
        itmx,
        ftol,
        reg_iters,
        psf_a.encode(),
        psf_b.encode(),
        iters,
        device_num,
        saves,
        output_bit,
        records,
        unmatched,
        psf_a_bp.encode(),
        psf_b_bp.encode(),
    )

    if not len(os.listdir(out_path)):
        os.removedirs(out_path)

    return list(records), itmx
