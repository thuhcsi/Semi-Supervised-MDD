import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", default=".", type=str, metavar="DIR", help="data directory"
    )

    return parser


def main(args):

    with open(os.path.join(args.data_dir, f"dict.wrd.txt"), "w") as f:

        lines = []
        with open(os.path.join(args.data_dir, 'train.wrd'), "r") as label_f:
            lines += label_f.readlines()
        with open(os.path.join(args.data_dir, 'valid.wrd'), "r") as label_f:
            lines += label_f.readlines()
        with open(os.path.join(args.data_dir, 'test.wrd'), "r") as label_f:
            lines += label_f.readlines()
        
        d = dict()
        for line in lines:
            for phn in line.split(' '):
                phn_ = phn.strip()
                if phn_ in d:
                    d[phn_] += 1
                else:
                    d[phn_] = 1
        for k, v in d.items():
            print(f'{k} {v}', file=f)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
