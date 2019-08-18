import argparse
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(
        os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


def is_valid_folder(parser, arg):
    if not os.path.exists(arg):
        parser.error("The folder %s does not exist!" % arg)
    if not os.path.isdir(arg):
        parser.error("%s is not a folder!" % arg)
    return arg


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def crop(args):
    if args.execute:
        jobsdir = os.path.join(args.folder, "_jobs")
        if not os.path.exists(jobsdir):
            jobsdir = input("Jobs not autodetected, enter directory: ")
        if os.path.exists(jobsdir):
            from gpufuse import execute
            gpufuse.execute(jobsdir, args.p, args.t)
        else:
            print("exiting")
    else:
        from gpufuse.crop import prep_experiment
        prep_experiment(args.folder, positions=args.p, tind=args.sum)


def fuse_in_mem(args):
    import concurrent.futures
    from gpufuse import crop

    jobs = crop.gather_jobs(args.folder)
    try:
        raise ImportError()
        from gpufuse.func import dev_info
        gpus = dev_info()
        minmem = 8500
        gpus = dict(filter(lambda x: int(x[1]["mem"]) > minmem, gpus.items()))
        if len(gpus) > 1:
            # add gpu device number to job
            for n, job in enumerate(jobs):
                job.append(n % len(gpus))
    except ImportError:
        gpus = [0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        ex.map(crop.crop_array_and_write, jobs)
    # return futures


def fuse(args):
    spim_a_folder = os.path.join(args.folder, "SPIMA")
    spim_b_folder = os.path.join(args.folder, "SPIMB")
    if not os.path.exists(spim_a_folder):
        spim_a_folder = input(
            "Could not autodetect SPIMA folder, enter SPIMA directory: "
        )
    if not os.path.exists(spim_b_folder):
        spim_b_folder = input(
            "Could not autodetect SPIMB folder, enter SPIMB directory: "
        )
    assert os.path.exists(spim_a_folder) and os.path.exists(
        spim_b_folder), "No folder"

    gpufuse.fusion_dualview_batch(
        spim_a_folder, args.PSF_A, spim_b_folder=spim_b_folder
    )


def view(args):
    import tifffile as tf
    import matplotlib.pyplot as plt

    if args.file.endswith(".tif") or args.file.endswith(".tiff"):
        print(f"loading {args.file}")
        im = tf.imread(args.file)

    elif os.path.isdir(args.file):
        from glob import glob

        im0 = glob(os.path.join(args.file, "**",
                                "*{:04d}*Pos0*.tif".format(args.t)))[0]
        print("loading timepoint {}, pos {}".format(args.t, args.p))
        im = tf.imread(im0, series=args.p)

    if args.max:
        tf.imshow(
            im.max(0),
            vmax=im.max() * args.contrast,
            cmap="gray",
            photometric="minisblack",
        )
    else:
        tf.imshow(
            im, vmax=im.max() * args.contrast, cmap="gray", photometric="minisblack"
        )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        'gpufuse', description="diSPIM fusion helper")
    subparsers = parser.add_subparsers(help="sub-commands")

    parser_fuse = subparsers.add_parser(
        "fuse", help="fuse prepared SPIMA/B folders")
    parser_fuse.add_argument(
        "folder",
        help="top level folder, containing SPIMA & SPIMB folders",
        type=lambda x: is_valid_folder(parser, x),
    )
    parser_fuse.add_argument(
        "--psfa",
        required=False,
        nargs=1,
        help="PSF for pathA",
        type=lambda x: is_valid_file(parser, x),
    )
    parser_fuse.add_argument(
        "-m", "--merge", action="store_true", help="merge channels after fusion"
    )
    parser_fuse.add_argument(
        "-r",
        "--reprocess",
        action="store_true",
        help="reprocess already fused images (otherwise skip)",
    )
    parser_fuse.add_argument(
        "-p",
        metavar="positions",
        type=int,
        nargs="+",
        help="specific positions to process, sep by spaces",
    )
    parser_fuse.add_argument(
        "-t",
        metavar="timepoints",
        type=int,
        nargs="+",
        help="specific timepoints to process, sep by spaces",
    )
    parser_fuse.add_argument(
        "--minfail",
        metavar="minfail",
        type=int,
        default=2,
        help="minimum number of failures before ignoring a position",
    )

    parser_crop = subparsers.add_parser(
        "crop", help="crop OME-formatted dispim series")
    parser_crop.add_argument(
        "folder",
        help="experiment folder with multiple timepoints",
        type=lambda x: is_valid_folder(parser, x),
    )
    parser_crop.add_argument(
        "-x", "--execute", action="store_true", help="run cropping jobs"
    )
    parser_crop.add_argument(
        "-p",
        metavar="positions",
        type=int,
        nargs="+",
        help="specific positions to process, sep by spaces",
    )
    parser_crop.add_argument(
        "-t",
        metavar="timepoints",
        type=int,
        nargs="+",
        help="specific timepoints to process, sep by spaces",
    )
    parser_crop.add_argument(
        "-s",
        "--sum",
        metavar="sumindices",
        type=int,
        nargs="+",
        default=[1, -1],
        help="time indices to sum when cropping, negative indices are relative to last timepoint",
    )

    parser_view = subparsers.add_parser(
        "view", help="View tiff file or experiment")
    parser_view.add_argument(
        "file",
        help="experiment file or folder with multiple timepoints",
        type=lambda x: is_valid_file(parser, x),
    )
    parser_view.add_argument(
        "-m", "--max", action="store_true", help="show max projection"
    )
    parser_view.add_argument(
        "--contrast",
        type=float,
        default=0.5,
        help="contrast of image (vmax = im.max * contrast)",
    )
    parser_view.add_argument(
        "-p",
        metavar="position",
        type=int,
        default=0,
        help="when path is an experiment, position to show",
    )
    parser_view.add_argument(
        "-t",
        metavar="timepoint",
        type=int,
        default=0,
        help="when path is an experiment, timepoint to show",
    )

    parser_fuse.set_defaults(func=fuse_in_mem)
    parser_crop.set_defaults(func=crop)
    parser_view.set_defaults(func=view)
    parser.set_defaults(func=lambda x: parser.print_help(sys.stderr))
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
