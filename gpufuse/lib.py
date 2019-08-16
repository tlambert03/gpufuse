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


LIBFILE = os.path.join(
    os.path.dirname(__file__), "bin", "libaim" + EXT
)
try:
    LIB = ctypes.CDLL(LIBFILE)
except OSError as e:
    print("Could not load gpufuse shared library: {}".format(e))
    LIB = None

if LIB:
    print(LIB)
    LIB.gettifinfo.restype = ctypes.c_ushort
    LIB.gettifinfo.argtypes = [ctypes.c_char_p, ctypes.c_uint * 3]

    LIB.readtifstack.restype = ctypes.c_void_p
    LIB.readtifstack.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_char_p,
        ctypes.c_uint * 3,
    ]

    LIB.reg_3dcpu.restype = ctypes.c_int
    LIB.reg_3dcpu.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_reg
        (ctypes.c_float * 16),  # float *iTmx
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_img1
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_img2
        (ctypes.c_uint * 3),  # unsigned int *imSize1
        (ctypes.c_uint * 3),  # unsigned int *imSize2
        ctypes.c_int,  # int regMethod
        ctypes.c_int,  # int inputTmx
        ctypes.c_float,  # float FTOL
        ctypes.c_int,  # int itLimit
        ctypes.c_int,  # int subBgTrigger
        (ctypes.c_float * 11),  # float *regRecords
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

    LIB.decon_singleview.restype = ctypes.c_int
    LIB.decon_singleview.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_decon
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_img
        (ctypes.c_uint * 3),  # unsigned int *imSize
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_psf
        (ctypes.c_uint * 3),  # unsigned int *psfSize
        ctypes.c_int,  # int itNumForDecon
        ctypes.c_int,  # int deviceNum
        ctypes.c_int,  # int gpuMemMode
        (ctypes.c_float * 10),  # float *deconRecord
        ctypes.c_bool,  # bool flagUnmatch
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_psf_bp
    ]

    # 3D registration and joint RL deconvolution with GPU implementation,
    # compatible with unmatched back projector.
    LIB.fusion_dualview.restype = ctypes.c_int
    LIB.fusion_dualview.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_decon,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_reg,
        (ctypes.c_float * 16),  # float *iTmx,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_img1,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_img2,
        (ctypes.c_uint * 3),  # unsigned int *imSizeIn1,
        (ctypes.c_uint * 3),  # unsigned int *imSizeIn2,
        (ctypes.c_float * 3),  # float *pixelSize1,
        (ctypes.c_float * 3),  # float *pixelSize2,
        ctypes.c_uint,  # int imRotation,
        ctypes.c_uint,  # int regMethod,
        ctypes.c_uint,  # int flagInitialTmx,
        ctypes.c_float,  # float FTOL,
        ctypes.c_uint,  # int itLimit,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_psf1,
        np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # float *h_psf2,
        (ctypes.c_uint * 3),  # unsigned int *psfSizeIn,
        ctypes.c_uint,  # int itNumForDecon,
        ctypes.c_uint,  # int deviceNum,
        ctypes.c_uint,  # int gpuMemMode,
        (ctypes.c_float * 22),  # float *fusionRecords,
        ctypes.c_bool,  # bool flagUnmatach,
        np.ctypeslib.ndpointer(
            ctypes.c_float, flags="C_CONTIGUOUS"
        ),  # float *h_psf_bp1,
        np.ctypeslib.ndpointer(
            ctypes.c_float, flags="C_CONTIGUOUS"
        ),  # float *h_psf_bp2
    ]

    # 3D registration and joint RL deconvolution with GPU implementation,
    # compatible with unmatched back projector. Processing for time-sequence images
    # in batch mode and APIs set to be File I/O from disk.
    # 3D registration and joint RL deconvolution with GPU implementation,
    # compatible with unmatched back projector.
    LIB.fusion_dualview_batch.restype = ctypes.c_int
    LIB.fusion_dualview_batch.argtypes = [
        # char *outFolder: directory which contains all output results
        ctypes.c_char_p,
        # char *inFolder1: directory for input image 1
        #                  if inFolder1 = “1”, switch to multiple color processing;
        ctypes.c_char_p,
        # char *inFolder2: directory for input image 2
        #                  if inFolder1 = “1”, set as main input folder (Fig 1)
        ctypes.c_char_p,
        # char *fileNamePrefix1: prefix of file names of image 1
        ctypes.c_char_p,
        # char *fileNamePrefix2: prefix of file names of image 2
        ctypes.c_char_p,
        # int imgNumStart: start time point
        ctypes.c_int,
        # int imgNumEnd: end time point
        ctypes.c_int,
        # int imgNumInterval: time point interval
        ctypes.c_int,
        # int imgNumTest: time point for test (used only if regMode = 1)
        ctypes.c_int,
        # float *pixelSize1: 3-element array to specify the pixel size (x, y, z) of image 1
        (ctypes.c_float * 3),
        # float *pixelSize2: 3-element array to specify the pixel size (x, y, z) of image 2
        (ctypes.c_float * 3),
        # int regMode: registration mode for batch processing:
        #   0: no registration, but perform ration, interpolation and transformation
        #      (based on initial matrix) on image 2;
        #   1: one image only, use test time point images to do registration and apply
        #      to all other time points
        #   2: all images independently, do registration for all images independently
        #   3: all images dependently, do registration for all images based on previous results
        ctypes.c_int,
        # int imRotation: rotate image 2 before registration
        #   0: no rotation
        #   1: 90deg rotation by y axis
        #   -1: -90deg rotation by y axis
        ctypes.c_int,
        # int flagInitialTmx: how to initialize the registration matrix
        #   0: default matrix
        #   1: use input matrix
        #   2: do translation registration based on phase information
        #   3: do translation registration based on 2D max projections
        ctypes.c_int,
        # float *iTmx: initial transformation matrix (used only if flagInitialTmx = 1)
        (ctypes.c_float * 16),
        # float FTOL: convergence threshold, defined as the difference of the cost function
        #             value from two adjacent iterations
        ctypes.c_float,
        # int itLimit: maximum iteration number
        ctypes.c_int,
        # char *filePSF1: forward projector 1 (PSF 1); should have isotropic pixel size
        ctypes.c_char_p,
        # char *filePSF2: forward projector 2 (PSF 2); should have isotropic pixel size
        ctypes.c_char_p,
        # int itNumForDecon: iteration number for deconvolution
        ctypes.c_int,
        # int deviceNum: the GPU device to be used, 0-based naming convention
        ctypes.c_int,
        # int *flagSaveInterFiles: 8-element array, save registered and max projection images
        #                          (1=yes, 0=no)
        #   [0]: Intermediate outputs
        #   [1]: reg A
        #   [2]: reg B
        #   [3]- [5]: Decon max projections Z, Y, X
        #   [6], [7]: Decon 3D max projections: Y, X
        (ctypes.c_uint * 8),
        # int bitPerSample: TIFF bit for deconvolution images
        ctypes.c_int,
        # float *records: 22-element array, records and feedbacks of the processing
        #   [0]-[10]: corresponding to regRecords for registration
        #   [11]-[20]: corresponding to deconRecords for deconvolution
        #   [21]: total time cost.
        (ctypes.c_float * 22),
        # bool flagUnmatch: use traditional back projectors or unmatched back projectors
        #   0: traditional
        #   1: unmatched
        ctypes.c_bool,
        # char *filePSF_bp1: unmatched back projector corresponding to h_psf1
        ctypes.c_char_p,
        # char *filePSF_bp2: unmatched back projector correspoånding to h_psf2
        ctypes.c_char_p,
    ]

    # Fig 1.  Folder convention for organizing multicolor datasets when using
    # fusion_dualview_batch” function. The “xxx ( )” indicates the name for the folders.
    #
    # xxx (main folder)/
    # ├── xxx (color1)/
    # │   ├── SPIMA
    # │   └── SPIMB
    # ├── xxx (color2)/
    # │   ├── SPIMA
    # │   └── SPIMB
    # └── xxx (... colorn)/
    #     ├── SPIMA
    #     └── SPIMB
