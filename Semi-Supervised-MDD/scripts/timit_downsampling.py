import os
import argparse
import subprocess


from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Downsampling L2-Arctic wav files")
parser.add_argument("--raw_l2_arctic_dir", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)


def main(args):
    subprocess.run(["cp", "-r", args.raw_l2_arctic_dir, args.output_dir])

    wav_list = glob(os.path.join(args.output_dir, "*/wav/*.wav"))

    for path in tqdm(wav_list, desc="downsampling L2-Arctic"):
        os.system(f'sox {path} -r 16000 ./tmp.WAV')
        os.system(f'mv ./tmp.WAV {path}')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)