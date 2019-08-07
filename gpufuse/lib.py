"""ctypes definitions for libaim.dll/so/dylib
"""
import ctypes
import os
import sys

import numpy as np

PLAT = sys.platform
if PLAT == "linux2":
    PLAT = "linux"
elif PLAT == "cygwin":
    PLAT = "win32"

EXT = ".dll"
if PLAT == "darwin":
    EXT = ".dylib"
elif PLAT == "linux":
    EXT = ".so"


LIBFILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "libaim" + EXT)
LIB = ctypes.CDLL(LIBFILE)

LIB.gettifinfo.restype = ctypes.c_ushort
LIB.gettifinfo.argtypes = [ctypes.c_char_p, ctypes.c_uint * 3]

LIB.readtifstack.restype = ctypes.c_void_p
LIB.readtifstack.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_char_p,
    ctypes.c_uint * 3,
]

LIB.reg_3dgpu.restype = ctypes.c_int
LIB.reg_3dgpu.argtypes = [
    # h_reg: registered image, it is the same size with the target image h_img1;
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # iTmx: 16-element array, output transformation matrix; if inputTmx = 1,
    # it is also used as the input for the initial transformation matrix;
    (ctypes.c_float * 16),
    # float *h_img1: target image, it is fixed
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # float *h_img2: source image, it is to be transformed to match image h_img1
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # unsigned int *imSize1: 3-element array to specify the image size (x, y, z) of h_img1
    (ctypes.c_uint * 3),
    # unsigned int *imSize2: 3-element array to specify the image size (x, y, z) of h_img2
    (ctypes.c_uint * 3),
    # int regMethod:
    #   0: no registration, transform image 2 based on input matrix
    #   1: translation only
    #   2: rigid body
    #   3: 7 degrees of freedom (translation, rotation, scaling equally in 3 dimensions)
    #   4: 9 degrees of freedom (translation, rotation, scaling)
    #   5: 12 degrees of freedom
    #   6: rigid body first, then do 12 degrees of freedom
    #   7: 3 DOF --> 6 DOF --> 9 DOF --> 12 DOF
    ctypes.c_int,  # int regMethod
    # inputTmx:
    #   0: use default initial matrix (h_img1 and h_img2 are aligned at center)
    #   1: use iTmx as initial matrix
    #   2: do translation registration based on phase information
    #   3: do translation registration based on 2D max projections
    #      (there is a bug with this option).
    ctypes.c_int,  # int inputTmx
    # FTOL: convergence threshold, defined as the difference of the cost function
    # value from two adjacent iterations
    ctypes.c_float,  # float FTOL
    ctypes.c_int,  # int itLimit: maximum iteration number;
    # flagSubBg: subject background before registration or not
    # (output image is still with background)
    #   0: no
    #   1: yes.
    ctypes.c_int,  # int flagSubBg
    # deviceNum: the GPU device to be used, 0-based naming convention.
    ctypes.c_int,  # int deviceNum
    # regRecords: 11-element array, records and feedbacks of the processing
    #   [0]-[3]:  initial GPU memory, after variables allocated, after processing,
    #             after variables released (all in MB)
    #   [4]-[6]:  initial cost function value, minimized cost function value,
    #             intermediate cost function value
    #   [7]-[10]: registration time (s), whole time (s), single sub iteration time (ms),
    #             total sub iterations
    (ctypes.c_float * 11),
]

LIB.decon_dualview.restype = ctypes.c_int
LIB.decon_dualview.argtypes = [
    # float *h_decon: deconvolved image, it is the same size with the input image h_img1
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # float *h_img1: input image 1
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # float *h_img2: input image 2
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # unsigned int *imSize: 3-element array to specify the image size (x, y, z)
    # of h_img1 or h_img2
    (ctypes.c_uint * 3),
    # float *h_psf1: forward projector 1 (PSF 1)
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # float *h_psf2: forward projector 2 (PSF 2)
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # unsigned int *psfSize: 3-element array to specify the image size (x, y, z)
    # of h_psf1 or h_psf2
    (ctypes.c_uint * 3),
    # int itNumForDecon: iteration number for deconvolution
    ctypes.c_int,
    # int deviceNum: the GPU device to be used, 0-based naming convention
    ctypes.c_int,
    # int gpuMemMode: the GPU memory mode
    #   0: Automatically set memory mode based on calculations
    #   1: sufficient memory
    #   2: memory optimized
    ctypes.c_int,
    # float *deconRecords: 10-element array, records and feedbacks of the processing
    #   [0]: the actual memory mode used;
    #   [1]-[5]: initial GPU memory, after variables partially allocated,
    #            during processing, after processing, after variables released (all in MB)
    #   [6]-[9]: initializing time, prepocessing time, decon time, total time
    (ctypes.c_float * 10),
    # bool flagUnmatch: use traditional back projectors or unmatched back projectors
    #   0: traditional
    #   1: unmatched
    ctypes.c_bool,
    # float *h_psf_bp1: unmatched back projector corresponding to h_psf1
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    # float *h_psf_bp2: unmatched back projector corresponding to h_psf2
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
]
