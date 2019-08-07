import json
import logging
import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from matplotlib.widgets import RectangleSelector
from scipy.interpolate import interp1d
from skimage.transform import resize
from multiprocessing.pool import Pool

tifffile_logger = logging.getLogger("tifffile")
tifffile_logger.setLevel(logging.ERROR)

POSITIONS = {x: i for i, x in enumerate("xyz")}


def determineThreshold(array, maxSamples=50000):
    array = np.array(array)
    elements = len(array)

    if elements > maxSamples:  # subsample
        step = round(elements / maxSamples)
        array = array[0::step]
        elements = len(array)

    connectingline = np.linspace(array[0], array[-1], elements)
    distances = np.abs(array - connectingline)
    position = np.argmax(distances)

    threshold = array[position]
    if np.isnan(threshold):
        threshold = 0
    return threshold


def selectiveMedianFilter(
    stack,
    backgroundValue=0,
    medianRange=3,
    verbose=False,
    withMean=False,
    deviationThreshold=None,
):
    """correct bad pixels on sCMOS camera.
    based on MATLAB code by Philipp J. Keller,
    HHMI/Janelia Research Campus, 2011-2014

    """
    from scipy.ndimage.filters import median_filter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        devProj = np.std(stack, 0, ddof=1)
        devProjMedFiltered = median_filter(devProj, medianRange, mode="constant")
        deviationDistances = np.abs(devProj - devProjMedFiltered)
        deviationDistances[deviationDistances == np.inf] = 0
        if deviationThreshold is None:
            deviationThreshold = determineThreshold(
                sorted(deviationDistances.flatten())
            )

        deviationMatrix = deviationDistances > deviationThreshold

        if withMean:
            meanProj = np.mean(stack, 0) - backgroundValue
            meanProjMedFiltered = median_filter(meanProj, medianRange)
            meanDistances = np.abs(meanProj - meanProjMedFiltered / meanProjMedFiltered)
            meanDistances[meanDistances == np.inf] = 0
            meanThreshold = determineThreshold(sorted(meanDistances.flatten()))

            meanMatrix = meanDistances > meanThreshold

            pixelMatrix = deviationMatrix | meanMatrix
            pixelCorrection = [
                deviationDistances,
                deviationThreshold,
                meanDistances,
                meanThreshold,
            ]
        else:
            pixelMatrix = deviationMatrix
            pixelCorrection = [deviationDistances, deviationThreshold]

        if verbose:
            pixpercent = (
                100 * np.sum(pixelMatrix.flatten()) / float(len(pixelMatrix.flatten()))
            )
            print(
                "Bad pixels detected: {} {:0.2f}".format(
                    np.sum(pixelMatrix.flatten()), pixpercent
                )
            )

        dt = stack.dtype
        out = np.zeros(stack.shape, dt)
        # apply pixelMatrix to correct insensitive pixels
        for z in range(stack.shape[0]):
            frame = np.asarray(stack[z], "Float32")
            filteredFrame = median_filter(frame, medianRange)
            frame[pixelMatrix == 1] = filteredFrame[pixelMatrix == 1]
            out[z] = np.asarray(frame, dt)

        return out, pixelCorrection


class Selection3D:
    def __init__(self):
        # keys are axis, values are array of (selector, axis_number) tuples
        self.selectors = []
        self.coords = {}
        self.linked_axes = []
        self.data_lim = [None, None, None]  # x, y, z Data Size

    def width(self, axis):
        if axis not in self.coords:
            raise ValueError("Unknown axis: %s" % axis)
        return round(np.diff(self.coords[axis])[0])

    def set_width(self, axis, value):
        # FIXME: improve this
        # using selector.ax.dataLim
        if axis not in self.coords:
            raise ValueError("Unknown axis: %s" % axis)
        self.coords[axis][1] = self.coords[axis][0] + value

    def _update_data_lim(self, mpl_ax, axes):
        # store the size of the underlying data
        for n, ax in enumerate(axes):
            if n == 0:
                size = mpl_ax.dataLim.width
            else:
                size = mpl_ax.dataLim.height
            self.data_lim[POSITIONS[ax]] = size

    def add_selector(self, mpl_ax, axes):
        self._update_data_lim(mpl_ax, axes)
        for ax in axes:
            if ax not in self.coords:
                self.coords[ax] = [0, 200]
        selector = RectangleSelector(
            mpl_ax,
            self.make_callback(axes),
            drawtype="box",
            useblit=False,
            rectprops=dict(facecolor="red", edgecolor="red", alpha=0.4, fill=False),
            interactive=True,
        )
        self.selectors.append((selector, axes))

    def make_callback(self, axes):
        def callback(eclick, erelease):
            x1, x2 = eclick.xdata, erelease.xdata
            y1, y2 = eclick.ydata, erelease.ydata
            width = np.abs(x1 - x2)
            height = np.abs(y1 - y2)
            for (myaxis, otheraxis, otherselector) in self.linked_axes:
                if not myaxis in "xy":
                    continue
                othermax = otherselector.data_lim[POSITIONS[otheraxis]]
                if myaxis == "x":
                    if width > othermax:
                        x2 = x1 + othermax
                if myaxis == "y":
                    if height > othermax:
                        y2 = y1 + othermax
                # if myaxis == 'x':
                #     for selector, axes in otherselector.selectors:
                #         if axes[0] == 'otheraxis':
                #             print(selector.ax.dataLim.width)
                #             if width > selector.ax.dataLim.width:
                #                 print("BIG")
                # if myaxis == 'y':
                #     for selector, axes in otherselector.selectors:
                #         if axes[1] == 'otheraxis':
                #             if height > selector.ax.dataLim.height:
                #                 print("BIG height")
            self.coords[axes[0]][0] = np.min([x1, x2])
            self.coords[axes[0]][1] = np.max([x1, x2])
            self.coords[axes[1]][0] = np.min([y1, y2])
            self.coords[axes[1]][1] = np.max([y1, y2])
            self.update()

        return callback

    def link_axis(self, myaxis, otheraxis, otherselector):
        if (myaxis, otheraxis, otherselector) not in self.linked_axes:
            self.linked_axes.append((myaxis, otheraxis, otherselector))
            otherselector.link_axis(otheraxis, myaxis, self)

    def update(self, links=True):
        for selector, axes in self.selectors:
            # x1, x2, y1, y2
            _extents = list(selector.extents)
            _extents[0:2] = self.coords[axes[0]]
            _extents[2:4] = self.coords[axes[1]]
            selector.extents = _extents
        if links:
            for (myaxis, otheraxis, otherselector) in self.linked_axes:
                if self.width(myaxis) != otherselector.width(otheraxis):
                    otherselector.set_width(otheraxis, self.width(myaxis))
            for sel in set([s[2] for s in self.linked_axes]):
                sel.update(links=False)


# def toggle_selector(event):
#     print(" Key pressed.")
#     if event.key in ["D", "d"] and toggle_selector.RS1.active:
#         print(" RectangleSelector deactivated.")
#         toggle_selector.RS1.set_active(False)
#     if event.key in ["A", "a"] and not toggle_selector.RS1.active:
#         print(" RectangleSelector activated.")
#         toggle_selector.RS1.set_active(True)


def parse_dispim_meta(impath):
    # parse TiffFile.ome_metadata to extract diSPIM info
    with tf.TiffFile(impath) as t:
        meta = t.ome_metadata
        # tifffile version > 2019.2.22
        if isinstance(meta, str):
            meta = tf.xml2dict(meta)["OME"]
    im0 = meta["Image"][0]
    out = {"channels": [], "nS": len(meta["Image"])}
    for x in "XYZCT":
        out["n" + x] = im0["Pixels"]["Size" + x]
    for x in "XYZ":
        out["d" + x] = im0["Pixels"]["PhysicalSize" + x]
    for chan in im0["Pixels"]["Channel"]:
        out["channels"].append(chan["Name"])
    out["dzRatio"] = out["dZ"] / out["dX"]
    return out


def get_exp_meta(exp):
    dirs = sorted(
        [d for d in os.listdir(exp) if not (d.startswith(".") or d.startswith("_"))]
    )
    im0 = sorted(glob(os.path.join(exp, dirs[0] + "/*.tif")))[0]
    meta = parse_dispim_meta(im0)
    meta["nT"] = len(dirs)
    meta["ind_a"] = [x for x, c in enumerate(meta["channels"]) if "RightCam" in c]
    meta["ind_b"] = [x for x, c in enumerate(meta["channels"]) if "LeftCam" in c]
    return meta


def load_dispim_mips(exp, tind=[0, -1], pos=0):
    """Get 3D stacks for t=indices"""
    meta = get_exp_meta(exp)

    for i, t in enumerate(tind):
        if t >= 0:
            idx = t
        else:
            idx = meta["nT"] + t

        im0 = glob(os.path.join(exp, "**", "*{:04d}*Pos0*.tif".format(idx)))[0]
        print("reading timepoint index {}".format(t))
        data = tf.imread(im0, series=pos)
        patha = data[:, meta["ind_a"]]
        pathb = data[:, meta["ind_b"]]
        if data.ndim == 4:
            _mip_xy = [patha.max(0).sum(0), pathb.max(0).sum(0)]
            _mip_xz = [patha.max(2).sum(1), pathb.max(2).sum(1)]
            _mip_zy = [patha.max(3).sum(1).T, pathb.max(3).sum(1).T]
        elif data.ndim == 3:
            _mip_xy = [patha.max(0), pathb.max(0)]
            _mip_xz = [patha.max(1), pathb.max(1)]
            _mip_zy = [patha.max(2).T, pathb.max(2).T]
        if i == 0:
            mip_xy = np.stack(_mip_xy).astype("single")
            mip_xz = np.stack(_mip_xz).astype("single")
            mip_zy = np.stack(_mip_zy).astype("single")
        else:
            mip_xy += np.stack(_mip_xy).astype("single")
            mip_xz += np.stack(_mip_xz).astype("single")
            mip_zy += np.stack(_mip_zy).astype("single")

    newshape = list(mip_zy.shape)
    newshape[-1] = round(newshape[-1] * meta["dzRatio"])
    mip_zyr = resize(mip_zy, newshape)
    newshape = list(mip_xz.shape)
    newshape[-2] = round(newshape[-2] * meta["dzRatio"])
    mip_xzr = resize(mip_xz, newshape)
    patha, pathb = [None] * 3, [None] * 3
    patha[0], pathb[0] = mip_xy
    patha[1], pathb[1] = mip_xzr
    patha[2], pathb[2] = mip_zyr
    return patha, pathb


def select_volume(patha, pathb, axial="zy", contrast=0.8):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1, adjustable="box")
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3, adjustable="box")
    ax4 = fig.add_subplot(2, 2, 4, sharey=ax1)
    plt.setp(ax2.get_yaxis(), visible=False)
    ax1.imshow(patha[0], aspect="equal", vmax=patha[0].max() * contrast, cmap="gray")
    ax2.imshow(
        np.fliplr(patha[2]), aspect="equal", vmax=patha[2].max() * contrast, cmap="gray"
    )
    ax3.imshow(pathb[0], aspect="equal", vmax=pathb[0].max() * contrast, cmap="gray")
    ax4.imshow(pathb[2], aspect="equal", vmax=pathb[2].max() * contrast, cmap="gray")
    plt.tight_layout()
    s1 = Selection3D()
    s1.add_selector(ax1, "xy")
    s1.add_selector(ax2, "zy")
    s2 = Selection3D()
    s2.add_selector(ax3, "xy")
    s2.add_selector(ax4, "zy")

    s1.link_axis("x", "z", s2)
    s1.link_axis("y", "y", s2)
    s1.link_axis("z", "x", s2)
    plt.show(block=True)
    extent_a = [s1.coords["x"], s1.coords["y"], s1.coords["z"]]
    extent_b = [s2.coords["x"], s2.coords["y"], s2.coords["z"]]
    extent_a = [[int(round(a[0])), int(round(a[1]))] for a in extent_a]
    extent_b = [[int(round(b[0])), int(round(b[1]))] for b in extent_b]
    return extent_a, extent_b


def prep_experiment(exp):
    meta = get_exp_meta(exp)
    outdir = os.path.join(exp, "_jobs")
    os.makedirs(outdir, exist_ok=True)
    for pos in range(meta["nS"]):
        print("loading position {}".format(pos))
        try:
            mips = load_dispim_mips(exp, pos=pos)
            ext_a, ext_b = select_volume(*mips)
            with open(os.path.join(outdir, "pos{}.json".format(pos)), "w") as _file:
                json.dump({"data": [exp, ext_a, ext_b, pos]}, _file)
        except IndexError:
            break
    return outdir


def crop_array(
    exp,
    extent_a,
    extent_b,
    meta,
    time,
    pos,
    outdir,
    background=100,
    kind="cubic",
    mfilter=False,
):

    im0 = glob(os.path.join(exp, "**", "*{:04d}*Pos0*.tif".format(time)))[0]
    print("loading file {}, position {}".format(im0, pos))
    data = tf.imread(im0, series=pos).astype("float")
    assert data.ndim in (3, 4), "cannot process %sd images" % data.ndim

    # crop xy
    temp = data[:, meta["ind_a"], slice(*extent_a[1]), slice(*extent_a[0])]

    # interpolate and crop in Z
    print("interpolating...")
    nZ = temp.shape[0]
    x = np.arange(0, nZ)
    f = interp1d(x, temp, axis=0, kind=kind)
    xx = np.linspace(0, nZ - 1, int(nZ * meta["dzRatio"]))
    _ext = int(nZ * meta["dzRatio"]) - np.array(list(reversed(extent_a[2])))
    out = f(xx[slice(*_ext)])
    out -= background
    out[out < 0] = 0
    out = out.astype("uint16")
    print("saving...")
    if out.ndim == 4:
        for c in range(out.shape[1]):
            path = os.path.join(outdir, "ch{}".format(c), "StackA_{}.tif".format(time))
            imout = out[:, c, np.newaxis, :, :]
            if mfilter:
                imout = selectiveMedianFilter(imout)[0]
            tf.imsave(path, imout, imagej=True)
    else:
        path = os.path.join(outdir, "ch0", "StackA_{}.tif".format(time))
        tf.imsave(path, out[:, np.newaxis, :, :], imagej=True)

    print("interpolating...")
    temp = data[:, meta["ind_b"], slice(*extent_b[1]), slice(*extent_b[0])]
    f = interp1d(
        x,
        data[:, meta["ind_b"], slice(*extent_b[1]), slice(*extent_b[0])],
        axis=0,
        kind=kind,
    )
    xx = np.linspace(0, nZ - 1, int(nZ * meta["dzRatio"]))
    out = f(xx[slice(*extent_b[2])])
    out -= background
    out[out < 0] = 0
    out = out.astype("uint16")

    torder = [3, 1, 2, 0] if out.ndim == 4 else [3, 2, 1]
    out = np.flip(np.transpose(out, torder), axis=0)
    print("saving...")
    if out.ndim == 4:
        for c in range(out.shape[1]):
            path = os.path.join(outdir, "ch{}".format(c), "StackB_{}.tif".format(time))
            imout = out[:, c, np.newaxis, :, :]
            if mfilter:
                imout = selectiveMedianFilter(imout)[0]
            tf.imsave(path, imout, imagej=True)
    else:
        path = os.path.join(outdir, "ch0", "StackB_{}.tif".format(time))
        tf.imsave(path, out[:, np.newaxis, :, :], imagej=True)


def crop_all(exp, extent_a, extent_b, pos=0):
    meta = get_exp_meta(exp)
    outdir = os.path.join(exp, "_cropped", "Pos{}".format(pos))
    os.makedirs(outdir, exist_ok=True)
    for c in range(meta["nC"] // 2):
        os.makedirs(os.path.join(outdir, "ch{}".format(c)), exist_ok=True)

    jobs = [(exp, extent_a, extent_b, meta, t, pos, outdir) for t in range(meta["nT"])]
    return jobs


def starcrop(args):
    return crop_array(*args)


def execute(jobsdir):
    jobs = []
    for js in glob(os.path.join(jobsdir, "*.json")):
        with open(js, "r") as f:
            d = json.load(f)["data"]
            jobs.extend(crop_all(*d))
    p = Pool()
    p.map(starcrop, jobs)


def main(impath):
    print("loading tiff...")
    im = tf.imread(impath)
    print("projecting...")

    dZ = 3
    dXY = 1

    # imA = im[:, 2, :, :]
    imA = im
    mipsAxy = imA.max(0)
    mipsAxz = np.rot90(imA, axes=(0, 1)).max(0)
    mipsAxzr = resize(mipsAxz, ((mipsAxz.shape[0] * dZ) // dXY, mipsAxz.shape[1]))
    mipsAyz = np.rot90(imA, axes=(0, 2)).max(0)
    mipsAyzr = resize(mipsAyz, (mipsAyz.shape[0], (mipsAyz.shape[1] * dZ) // dXY))

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1, adjustable="box", aspect="equal")
    ax2 = fig.add_subplot(2, 2, 3, sharex=ax1)
    ax3 = fig.add_subplot(2, 2, 2, sharey=ax1)
    plt.setp(ax2.get_yaxis(), visible=False)
    plt.setp(ax3.get_xaxis(), visible=False)
    ax1.imshow(mipsAxy)
    ax2.imshow(mipsAxzr)
    ax3.imshow(mipsAyzr)
    plt.tight_layout()

    selection3d = Selection3D()
    selection3d.add_selector(ax1, "xy")
    selection3d.add_selector(ax2, "xz")
    selection3d.add_selector(ax3, "zy")

    # RS3 = make_selector(ax3)
    # toggle_selector.RS1 = RS1
    # plt.connect("key_press_event", toggle_selector)
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        jobsdir = sys.argv[1]
        execute(jobsdir)