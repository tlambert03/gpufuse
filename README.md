# gpufuse
Python wrapper for Min Guo's CUDA-based dual-view RL fusion 

```python

import gpufuse 
import tifffile

# register two 3D arrays on GPU
im_a = tifffile.imread("StackA_0.tif")
im_b = tifffile.imread("StackB_0.tif")
reg_b, tmx, record = gpufuse.reg_3dgpu(im_a, im_b)

# dual-view RL decon to fuse two registered arrays on GPU
psf_a = tifffile.imread("PSFA.tif")
psf_b = tifffile.imread("PSFB.tif")
deconvolved, record = gpufuse.decon_dualview(im_a, reg_b, psf_a, psf_b)

# joint fusion with a single helper function
deconvolved, reg_b = gpufuse.fuse(im_a, im_b, psf_a, psf_b)

```

## Requirements

This package assumes you have a CUDA-capable GPU, have installed the CUDA runtime libraries, and have cudart/cufft available in your library path.