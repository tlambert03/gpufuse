import json
import logging
import os
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from matplotlib.widgets import Button, RectangleSelector
from scipy.interpolate import interp1d

# from skimage.transform import resize
from .util import resize_axis

from .filter import selectiveMedianFilter

tifffile_logger = logging.getLogger("tifffile")
tifffile_logger.setLevel(logging.ERROR)

POSITIONS = {x: i for i, x in enumerate("xyz")}
__JOBSDIR__ = "_jobs"
__DECONDIR__ = "_decon"
__TMXDIR__ = "_tmx"


class Selection3D:
    def __init__(self, coords=None):
        # keys are axis, values are array of (selector, axis_number) tuples
        self.selectors = []
        self.coords = {}
        if coords is not None:
            self.coords = {"x": coords[0], "y": coords[1], "z": coords[2]}
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
            rectprops=dict(facecolor="red", edgecolor="red",
                           alpha=0.4, fill=False),
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
                if myaxis not in "xy":
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
    im0 = meta["Image"][0] if isinstance(
        meta["Image"], list) else meta["Image"]
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
        [d for d in os.listdir(exp) if not (
            d.startswith(".") or d.startswith("_"))]
    )
    try:
        im0 = sorted(glob(os.path.join(exp, dirs[0] + "/*.tif")))[0]
    except IndexError:
        raise FileNotFoundError("No tiff files found in {}".format(exp))
    meta = parse_dispim_meta(im0)
    meta["nT"] = len(dirs)
    meta["ind_a"] = [x for x, c in enumerate(
        meta["channels"]) if "RightCam" in c]
    meta["ind_b"] = [x for x, c in enumerate(
        meta["channels"]) if "LeftCam" in c]
    return meta


def load_dispim_mips(exp, tind=None, pos=0):
    """Get 3D stacks for t=indices"""
    meta = get_exp_meta(exp)

    if tind is None:
        tind = [0]

    for i, t in enumerate(tind):
        if t >= 0:
            idx = t
        else:
            idx = meta["nT"] + t

        im0 = glob(os.path.join(exp, "**", "*{:04d}*Pos0*.tif".format(idx)))[0]
        print("reading timepoint index {}, pos {}".format(t, pos))
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

    # newshape = list(mip_zy.shape)
    # newshape[-1] = round(newshape[-1] * meta["dzRatio"])
    # mip_zyr = skimage.transform.resize(mip_zy, newshape)
    mip_zyr = resize_axis(mip_zy, round(
        mip_zy.shape[-1] * meta["dzRatio"]), axis=-1)
    # newshape = list(mip_xz.shape)
    # newshape[-2] = round(newshape[-2] * meta["dzRatio"])
    # mip_xzr = skimage.transform.resize(mip_xz, newshape)
    mip_xzr = resize_axis(mip_xz, round(
        mip_xz.shape[-2] * meta["dzRatio"]), axis=-2)

    patha, pathb = [None] * 3, [None] * 3
    patha[0], pathb[0] = mip_xy
    patha[1], pathb[1] = mip_xzr
    patha[2], pathb[2] = mip_zyr
    return patha, pathb


def select_volume(patha, pathb, initial_coords=None, contrast=0.8, controller=None):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1, adjustable="box")
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3, adjustable="box")
    ax4 = fig.add_subplot(2, 2, 4, sharey=ax1)
    plt.setp(ax2.get_yaxis(), visible=False)
    ax1.imshow(patha[0], aspect="equal",
               vmax=patha[0].max() * contrast, cmap="gray")
    ax2.imshow(
        np.fliplr(patha[2]), aspect="equal", vmax=patha[2].max() * contrast, cmap="gray"
    )
    ax3.imshow(pathb[0], aspect="equal",
               vmax=pathb[0].max() * contrast, cmap="gray")
    ax4.imshow(pathb[2], aspect="equal",
               vmax=pathb[2].max() * contrast, cmap="gray")
    plt.tight_layout()
    s1 = Selection3D(
        coords=initial_coords[0] if initial_coords is not None else None)
    s2 = Selection3D(
        coords=initial_coords[1] if initial_coords is not None else None)
    s1.add_selector(ax1, "xy")
    s1.add_selector(ax2, "zy")
    s2.add_selector(ax3, "xy")
    s2.add_selector(ax4, "zy")

    s1.link_axis("x", "z", s2)
    s1.link_axis("y", "y", s2)
    s1.link_axis("z", "x", s2)
    if initial_coords is not None:
        s1.update(links=False)
        s2.update(links=False)

    if controller is not None:
        axprev = plt.axes([0.86, 0.32, 0.1, 0.075])
        axnext = plt.axes([0.86, 0.23, 0.1, 0.075])
        askip = plt.axes([0.86, 0.14, 0.1, 0.075])
        astop = plt.axes([0.86, 0.05, 0.1, 0.075])
        bprev = Button(axprev, "Previous")
        bprev.on_clicked(controller.prev)
        bnext = Button(axnext, "Next")
        bnext.on_clicked(controller.next)
        askip = Button(askip, "Skip")
        askip.on_clicked(controller.skip)
        astop = Button(astop, "Stop")
        astop.on_clicked(controller.stop)
        fig.canvas.set_window_title(
            "Position: {}".format(controller.positions[controller.i])
        )

    plt.show(block=True)
    extent_a = [s1.coords["x"], s1.coords["y"], s1.coords["z"]]
    extent_b = [s2.coords["x"], s2.coords["y"], s2.coords["z"]]
    extent_a = [[int(round(a[0])), int(round(a[1]))] for a in extent_a]
    extent_b = [[int(round(b[0])), int(round(b[1]))] for b in extent_b]
    return extent_a, extent_b


class Controller(object):
    def __init__(self, positions):
        self.positions = positions
        self.i = 0
        self.abort = False
        self.skipped = False

    def next(self, event):
        self.i += 1
        plt.close()

    def prev(self, event):
        self.i -= 1
        if self.i < 0:
            self.i = 0
        plt.close()

    def skip(self, event):
        self.i += 1
        self.skipped = True
        plt.close()

    def stop(self, event):
        self.abort = True
        plt.close()


def prep_experiment(exp, positions=None, tind=None):
    try:
        meta = get_exp_meta(exp)
    except FileNotFoundError as e:
        print(e)
        return
    outdir = os.path.join(exp, __JOBSDIR__)
    os.makedirs(outdir, exist_ok=True)

    controller = Controller(positions or range(meta["nS"]))
    while True:
        try:
            p = controller.positions[controller.i]
        except IndexError:
            break
        print("loading position {}".format(p))
        try:
            mips = load_dispim_mips(exp, pos=p, tind=tind)
            try:
                d = parse_job(os.path.join(outdir, "pos{}.json".format(p)))
                initial_coords = [d["extent_a"], d["extent_b"]]
                # old format
                # with open(os.path.join(outdir, "pos{}.json".format(p)), "r") as _file:
                #     d = json.load(_file)["data"]
                #     initial_coords = [d[1], d[2]]
            except Exception:
                initial_coords = None
            controller.skipped = False
            ext_a, ext_b = select_volume(
                *mips, initial_coords=initial_coords, controller=controller
            )
            if not controller.skipped:
                with open(os.path.join(outdir, "pos{}.json".format(p)), "w") as _file:
                    # json.dump({"data": [os.path.abspath(exp), ext_a, ext_b, p]}, _file)
                    json.dump(
                        {
                            "exp": os.path.abspath(exp),
                            "extent_a": ext_a,
                            "extent_b": ext_b,
                            "pos": p,
                        },
                        _file,
                    )

        except IndexError:
            break
        if controller.abort:
            break
    return outdir


def crop_array_inmem(
    exp, extent_a, extent_b, meta, time, pos, chan, device_num=0, background=100
):
    from .func import fusion_dualview

    im0 = glob(os.path.join(exp, "**", "*{:04d}*Pos0*.tif".format(time)))[0]
    print(
        f"loading file {os.path.basename(im0)}, pos {pos}, t: {time}, c: {chan}")
    maxn = meta["nC"] * meta["nZ"]
    keys = list(range(meta["ind_a"][chan], maxn, meta["nC"]))
    keys.extend(list(range(meta["ind_b"][chan], maxn, meta["nC"])))
    keys.sort()
    data = tf.imread(im0, series=pos, key=keys)
    data = data.reshape((meta["nZ"], 2, -1, data.shape[-1]))
    slc_a_x, slc_a_y, _slc_a_z = extent_a
    slc_b_x, slc_b_y, slc_b_z = extent_b
    # weird reversing is due to the way the slices were picked in select_volume()
    slc_a_z = [None, None]
    slc_a_z[0] = round((meta["dzRatio"] * meta["nZ"] -
                        _slc_a_z[1]) / meta["dzRatio"])
    slc_a_z[1] = round((meta["dzRatio"] * meta["nZ"] -
                        _slc_a_z[0]) / meta["dzRatio"])
    slc_b_z = [round(i / meta["dzRatio"]) for i in slc_b_z]
    idx_a = 0 if meta["ind_a"][chan] < meta["ind_b"][chan] else 1
    idx_b = 1 if idx_a == 0 else 0
    im_a = data[slice(*slc_a_z), idx_a, slice(*slc_a_y), slice(*slc_a_x)].astype(
        "single"
    )
    im_b = data[slice(*slc_b_z), idx_b, slice(*slc_b_y), slice(*slc_b_x)].astype(
        "single"
    )
    im_a -= background
    im_b -= background
    im_a[im_a < 0] = 0
    im_b[im_b < 0] = 0

    # look for existing tranformation matrix from same position
    tmx_mode = 0
    itmx = None
    try:
        tmxfile = glob(os.path.join(exp, __TMXDIR__, f"p{pos}*.txt"))[0]
        itmx = np.genfromtxt(tmxfile)
        tmx_mode = 1
        print('found existing tmx file')
    except Exception:
        pass

    return fusion_dualview(
        im_a,
        im_b,
        pixel_size1=[meta["dX"], meta["dX"], meta["dZ"]],
        rot_mode=1,
        device_num=device_num,
        itmx=itmx,
        tmx_mode=tmx_mode,
    )


def with_lock(func):
    def wrapper(args):
        exp, extent_a, extent_b, meta, time, pos, chan, *rest = args
        code = f"p{pos}_t{time}_c{chan}"
        outfolder = os.path.join(exp, __DECONDIR__)
        lockfile = os.path.join(outfolder, f'.{code}.lock')
        if not os.path.exists(lockfile):
            Path(lockfile).touch()
            try:
                val = func(args)
            except Exception as e:
                os.remove(lockfile)
                raise e
            else:
                os.remove(lockfile)
                return val
    return wrapper


@with_lock
def crop_array_and_write(args):
    # meant for parallel execution
    exp, extent_a, extent_b, meta, time, pos, chan, *rest = args
    gpu = args[7] if len(args) > 7 else 0

    outfolder = os.path.join(exp, __DECONDIR__)
    tmxfolder = os.path.join(exp, __TMXDIR__)

    code = f"p{pos}_t{time}_c{chan}"
    name = os.path.join(outfolder, f"{code}.tif")

    logfile = os.path.join(outfolder, 'log.json')
    if os.path.exists(name):
        info = {}
        if os.path.exists(logfile):
            with open(logfile, "r") as jsonFile:
                info = json.load(jsonFile)
        info['skipped'] = info.get('skipped', [])
        info['skipped'].append((code, 'already exists'))
        with open(logfile, "w") as jsonFile:
            json.dump(info, jsonFile)
        print(f"SKIPPING pos{pos} t{time} c{chan} - already exists")
        return {'err': False, 'skipped': True}
    if os.path.exists(logfile):
        with open(logfile, "r") as jsonFile:
            info = json.load(jsonFile)
        if 'errors' in info and f"gpu{gpu}_pos{pos}" in info['errors']:
            info['skipped'] = info.get('skipped', [])
            info['skipped'].append(
                (code, 'previous error'))
            with open(logfile, "w") as jsonFile:
                json.dump(info, jsonFile)
            print(f"SKIPPING pos{pos} on gpu {gpu} - previous errors")
            return {'err': False, 'skipped': True}

    try:
        h_decon, h_reg, records, itmx = crop_array_inmem(*args)
        os.makedirs(outfolder, exist_ok=True)
        print(f"saving {name}...")
        tf.imsave(
            name,
            h_decon[:, np.newaxis, :, :].astype("single"),
            imagej=True,
        )
        os.makedirs(tmxfolder, exist_ok=True)
        tmxname = os.path.join(tmxfolder, f"{code}.txt")
        np.savetxt(tmxname, np.array(list(itmx)), fmt='%.10f')
        return {'err': False, 'skipped': False, 'records': records, 'tmx': itmx}
    except Exception as e:
        import sys
        import traceback
        traceback.print_exc(file=sys.stdout)
        info = {}
        if os.path.exists(logfile):
            with open(logfile, "r") as jsonFile:
                info = json.load(jsonFile)
        info['errors'] = info.get('errors', {})
        info['errors'][f"gpu{gpu}_pos{pos}"] = str(e)
        with open(logfile, "w") as jsonFile:
            json.dump(info, jsonFile)

        return {'err': True, 'msg': str(e)}


def crop_array(
    exp,
    extent_a,
    extent_b,
    meta,
    time,
    pos,
    outdir,
    subfolders=True,
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
            subf = os.path.join(outdir, "ch{}".format(c))
            fname = "StackA_{}.tif".format(time)
            if subfolders:
                subf = os.path.join(subf, "SPIMA")
                os.makedirs(subf, exist_ok=True)
                fname = "SPIMA_{}.tif".format(time)
            path = os.path.join(subf, fname)
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
            subf = os.path.join(outdir, "ch{}".format(c))
            fname = "StackB_{}.tif".format(time)
            if subfolders:
                subf = os.path.join(subf, "SPIMB")
                os.makedirs(subf, exist_ok=True)
                fname = "SPIMB_{}.tif".format(time)
            path = os.path.join(subf, fname)
            imout = out[:, c, np.newaxis, :, :]
            if mfilter:
                imout = selectiveMedianFilter(imout)[0]
            tf.imsave(path, imout, imagej=True)
    else:
        path = os.path.join(outdir, "ch0", "StackB_{}.tif".format(time))
        tf.imsave(path, out[:, np.newaxis, :, :], imagej=True)


def starcrop(args):
    try:
        return crop_array(*args)
    except Exception:
        print("Job failed: {}".format(args))
        return None


def parse_job(filepath):
    with open(filepath, "r") as _file:
        j = json.load(_file)
    if "data" in j:
        # old format
        data = j.pop("data")
        j.update(
            {"exp": data[0], "extent_a": data[1],
                "extent_b": data[2], "pos": data[3]}
        )
    return j


def crop_all(exp, extent_a, extent_b, pos=0):
    meta = get_exp_meta(exp)
    # outdir = os.path.join(exp, "_cropped", "Pos{}".format(pos))
    # os.makedirs(outdir, exist_ok=True)
    # for c in range(meta["nC"] // 2):
    #     os.makedirs(os.path.join(outdir, "ch{}".format(c)), exist_ok=True)

    jobs = [(exp, extent_a, extent_b, meta, t, pos) for t in range(meta["nT"])]
    return jobs


def gather_jobs(jobsdir, positions=None, timepoints=None, channels=None):
    if isinstance(channels, int):
        channels = [channels]
    jobs = []
    jfiles = glob(os.path.join(jobsdir, "*.json"))
    if not jfiles:
        jfiles = glob(os.path.join(jobsdir, __JOBSDIR__, "*.json"))
    if not jfiles:
        print("No .json jobs found in {}".format(jobsdir))
        return jobs
    for js in jfiles:
        d = parse_job(js)
        if positions is not None:
            if d["pos"] not in positions:
                continue
        meta = get_exp_meta(d["exp"])
        channels = channels or range(meta["nC"] // 2)
        for t in range(meta["nT"]):
            for c in channels:
                # these will be the arguments for crop_inmem
                # (exp, extent_a, extent_b, meta, time, pos, chan=0, device_num=0)
                jobs.append(
                    [d["exp"], d["extent_a"], d["extent_b"], meta, t, d["pos"], c]
                )

    if timepoints is not None:
        jobs = [j for j in jobs if j[4] in timepoints]
    return jobs


def execute(jobsdir, positions=None, timepoints=None, client=None):
    jobs = gather_jobs(jobsdir, positions, timepoints)
    if client is None:
        client = Pool()
    assert hasattr(
        client, "map"), "provided client does not implement a map function"
    return client.map(starcrop, jobs)


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
    # mipsAxzr = skimage.transform.resize(
    #   mipsAxz, ((mipsAxz.shape[0] * dZ) // dXY, mipsAxz.shape[1]))
    mipsAxzr = resize_axis(mipsAxz, (mipsAxz.shape[0] * dZ) // dXY, axis=0)

    mipsAyz = np.rot90(imA, axes=(0, 2)).max(0)
    # mipsAyzr = skimage.transform.resize(
    #   mipsAyz, (mipsAyz.shape[0], (mipsAyz.shape[1] * dZ) // dXY))
    mipsAyzr = resize_axis(mipsAyz, (mipsAyz.shape[1] * dZ) // dXY, axis=1)

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
