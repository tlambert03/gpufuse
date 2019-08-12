import numpy as np


def gauss1d(x, sigma=1, mean=0, amp=1):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def gauss3d(mean=(0, 0, 0), sigma=(1, 1, 1), amp=1):
    def func(x, y, z):
        sigx, sigy, sigz = sigma
        meanx, meany, meanz = mean
        return amp * np.exp(
            -(
                (x - meanx) ** 2 / (2 * sigx ** 2)
                + (y - meany) ** 2 / (2 * sigy ** 2)
                + (z - meanz) ** 2 / (2 * sigz ** 2)
            )
        )

    return func


def gauss3d_kernel(shape=(100, 128, 128), sigx=1, sigz=2):
    f = gauss3d(mean=(0, 0, 0), sigma=(sigx, sigx, sigz))
    z, y, x = np.mgrid[
        -shape[0] // 2 : shape[0] // 2,
        -shape[1] // 2 : shape[1] // 2,
        -shape[2] // 2 : shape[2] // 2,
    ]
    a = f(x, y, z)
    return a


def gauss_psf(nxy=128, nz=128, dz=0.1625, dxy=0.1625, wvl=0.55, NA=0.8):
    sigx = ((0.61 * wvl / NA) / 2.355) / dxy
    sigz = ((2 * wvl / NA ** 2) / 2.355) / dz
    psf = gauss3d_kernel((nz, nxy, nxy), sigx, sigz)
    return psf


def real_psf(nxy=128, nz=128, dz=0.1625, dxy=0.1625, wvl=0.55, NA=0.8):
    import microscPSF.microscPSF as mpsf

    params = mpsf.m_params.copy()
    params.update(
        {
            "M": 40,
            "NA": NA,
            # "ng0": 1.33,  # coverslip RI design value
            # "ng": 1.33,
            # "ni0": 1.33,  # immersion medium RI design value
            # "ni0": 1.33,
            "tg": 0,
            "tg0": 0,  # microns, coverslip thickness design value
            "ti0": 3500,  # microns, working distance design value
            "zd0": 200.0 * 1.0e3,  # microscope tube length (in microns).
        }
    )
    zv = np.arange(-nz // 2, nz // 2) * dz
    psf = mpsf.gLXYZFocalScan(
        params, dxy, nxy, zv, normalize=True, pz=0.0, wvl=wvl, zd=None
    )
    return psf


def psf(real=True, **kwargs):
    if real:
        try:
            return real_psf(**kwargs)
        except ImportError:
            print("could not import microscPSF, falling back to 3D gaussian PSF")
            pass
    return gauss_psf(**kwargs)


def spim_psf(sheet_fwhm=3, real=True, **kwargs):
    _psf = psf(real, **kwargs)
    sheet_sig = (sheet_fwhm / 2.355) / kwargs.get("dz", 0.1625)
    z = np.arange(-_psf.shape[0] // 2, _psf.shape[0] // 2)
    sheet_profile = gauss1d(z, sheet_sig)
    newpsf = sheet_profile[:, np.newaxis, np.newaxis] * _psf
    return newpsf / newpsf.sum()
