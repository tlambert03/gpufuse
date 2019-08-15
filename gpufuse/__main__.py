import argparse
import os
import sys

def main():
    try:
        import gpufuse
    except ImportError:
        PACKAGE_PARENT = ".."
        SCRIPT_DIR = os.path.dirname(
            os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
        )
        sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
        import gpufuse


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
            print(jobsdir)
            if not os.path.exists(jobsdir):
                jobsdir = input("Jobs not autodetected, enter directory: ")
            if os.path.exists(jobsdir):
                gpufuse.execute(jobsdir, args.p, args.t)
            else:
                print("exiting")
        else:
            gpufuse.prep_experiment(args.folder, args.p)


    def fuse_in_thread(job, chan, name):
        from concurrent.futures import ProcessPoolExecutor, process
        with ProcessPoolExecutor() as executor:
            decon = executor.map(gpufuse.crop.starcrop_inmem, ((job, chan),))
        try:
            return list(decon)[0]
        except process.BrokenProcessPool:
            return None


    def fuse_in_mem(args):
        import tifffile as tf
        import numpy as np

        jobs = gpufuse.crop.gather_jobs(args.folder)
        meta = gpufuse.crop.get_exp_meta(args.folder)
        outfolder = os.path.join(args.folder, "_decon")
        os.makedirs(outfolder, exist_ok=True)
        
        for job in jobs:
            res = []
            t = job[4]
            pos = job[5]
            if args.p and pos not in args.p:
                continue
            if args.t and t not in args.t:
                continue
            if args.merge:
                name = os.path.join(outfolder, f"p{pos}_t{t}.tif")
                if os.path.exists(name) and not args.reprocess:
                    print(f"skipping already processed file {name}")
                    continue
                for chan in range(meta["nC"] // 2):
                    decon = fuse_in_thread(job, chan, name)
                    if decon is not None:
                        res.append(decon)
                if len(res):
                    tf.imsave(name, np.stack(res, 1).astype("single"), imagej=True)
            else:
                for chan in range(meta["nC"] // 2):
                    name = os.path.join(outfolder, f"p{pos}_t{t}_c{chan}.tif")
                    if os.path.exists(name) and not args.reprocess:
                        print(f"skipping already processed file {name}")
                        continue
                    decon = fuse_in_thread(job, chan, name)
                    if decon is not None:
                        tf.imsave(
                            name, decon[:, np.newaxis, :, :].astype("single"), imagej=True
                        )


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
        assert os.path.exists(spim_a_folder) and os.path.exists(spim_b_folder), "No folder"

        print(args.PSF_A)
        gpufuse.fusion_dualview_batch(
            spim_a_folder, args.PSF_A, spim_b_folder=spim_b_folder
        )


    parser = argparse.ArgumentParser(description="diSPIM fusion helper")
    subparsers = parser.add_subparsers(help="sub-commands")

    parser_fuse = subparsers.add_parser("fuse", help="fuse prepared SPIMA/B folders")
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


    parser_crop = subparsers.add_parser("crop", help="crop OME-formatted dispim series")
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

    parser_fuse.set_defaults(func=fuse_in_mem)
    parser_crop.set_defaults(func=crop)
    parser.set_defaults(func=lambda x: parser.print_help(sys.stderr))
    args = parser.parse_args()

    args.func(args)

if __name__ == "__main__":
    main()