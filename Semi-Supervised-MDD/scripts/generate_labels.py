import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--segment", default="train", type=str, help="train/valid/test"
    )
    parser.add_argument("--phn_text_path", required=True, type=str, help="path to phn label file")
    return parser


def main(args):

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)


    with open(os.path.join(args.dest, f"{args.segment}.wrd"), "w") as f:

        with open(args.phn_text_path, "r") as scp_f:
            lines = scp_f.readlines()
            for line in lines:
                labels = line.strip().split(' ', 1)[1]

                print(
                    labels, file=f
                )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
