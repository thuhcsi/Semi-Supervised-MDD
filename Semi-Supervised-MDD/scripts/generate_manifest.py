import argparse
import os

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--segment", default="train", type=str, help="train/valid/test"
    )
    parser.add_argument("--scp_path", required=True, type=str, help="path to kaldi wav.csp file")
    return parser


def main(args):

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)


    with open(os.path.join(args.dest, f"{args.segment}.tsv"), "w") as f:
        dir_path = os.path.realpath(args.root)

        print(dir_path, file=f)

        with open(args.scp_path, "r") as scp_f:
            lines = scp_f.readlines()
            for line in lines:
                fname = line.strip().split(' ')[1]
                file_path = os.path.realpath(fname)

                frames = soundfile.info(fname).frames

                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=f
                )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
