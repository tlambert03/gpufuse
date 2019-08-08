import argparse
import os
import sys

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
            gpufuse.execute(jobsdir)
        else:
            print("exiting")
    else:
        gpufuse.prep_experiment(args.folder, args.p)


def fuse(args):
    spim_a_folder = os.path.join(args.folder, 'SPIMA')
    spim_b_folder = os.path.join(args.folder, 'SPIMB')
    if not os.path.exists(spim_a_folder):
        spim_a_folder = input("Could not autodetect SPIMA folder, enter SPIMA directory: ")
    if not os.path.exists(spim_b_folder):
        spim_b_folder = input("Could not autodetect SPIMB folder, enter SPIMB directory: ")
    assert os.path.exists(spim_a_folder) and os.path.exists(spim_b_folder), 'No folder'
    
    print(args.PSF_A)
    gpufuse.fusion_dualview_batch(spim_a_folder, args.PSF_A, spim_b_folder=spim_b_folder)


parser = argparse.ArgumentParser(description="diSPIM fusion helper")
subparsers = parser.add_subparsers(help="sub-commands")

parser_fuse = subparsers.add_parser("fuse", help="fuse prepared SPIMA/B folders")
parser_fuse.add_argument(
    "folder",
    help="top level folder, containing SPIMA & SPIMB folders",
    type=lambda x: is_valid_folder(parser, x),
)
parser_fuse.add_argument(
    "PSF_A",
    help="PSF for pathA",
    type=lambda x: is_valid_file(parser, x),
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
    metavar="pos",
    type=int,
    nargs="+",
    help="specific positions to process, sep by spaces",
)

parser_fuse.set_defaults(func=fuse)
parser_crop.set_defaults(func=crop)
args = parser.parse_args()

args.func(args)
