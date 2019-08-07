# gpufuse
python wrapper for Min Guo's CUDA-based dual-view RL fusion 

```python

import gpufuse 

# register two 3D arrays on GPU
im_a = tf.imread("StackA_0.tif")
im_b = tf.imread("StackB_0.tif")
reg_b, tmx, record = gpufuse.reg_3dgpu(im_a, im_b)

# dual-view RL decon to fuse two registered arrays on GPU
psf_a = tf.imread("PSFA.tif")
psf_b = tf.imread("PSFB.tif")
deconvolved, record = gpufuse.decon_dualview(im_a, reg_b, psf_a, psf_b)

# joint fusion with a single helper function
deconvolved, reg_b = gpufuse.fuse(im_a, im_b, psf_a, psf_b)

```