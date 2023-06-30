import argparse
import os
import glob

import soundfile

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )

    parser.add_argument("--l2_path", required=True, type=str, help="l2-Arctic path")
    return parser


def main(args):

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    train_spk = ["MBMPS","ERMS","TLV","PNV","ASI","RRBI","TXHC","LXC","HJK","HKK","ABA","SKA"]

    unlabel_list = []
    for spk in train_spk:
        paths = os.path.join(args.l2_path, spk, 'annotation/*.TextGrid')
        anno_list = [os.path.split(p)[1].split('.')[0] for p in glob.glob(paths)]

        paths = os.path.join(args.l2_path, spk, 'wav/*.wav')
        for p in glob.glob(paths):
            id_ = os.path.split(p)[1].split('.')[0]
            if id_ not in anno_list:
                unlabel_list.append(p)

    with open(os.path.join(args.dest, f"train_u.tsv"), "w") as f:
        dir_path = os.path.realpath(args.root)

        print(dir_path, file=f)


        for file_path in tqdm(unlabel_list, desc='generating manifest for unlabeled data'):

            frames = soundfile.info(file_path).frames

            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=f
            )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
