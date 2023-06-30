import os
import sys
import string
import pathlib
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from g2p.en_us import G2P


train_spk = ["MBMPS","ERMS","TLV","PNV","ASI","RRBI","TXHC","LXC","HJK","HKK","ABA","SKA"]


parser = argparse.ArgumentParser(description="Prepare trans data")
parser.add_argument("--l2arctic_dir", default="/ceph/home/ywx20/data/L2_ARTIC_v5", help="l2-Arctic am output path")
parser.add_argument("--save_path", default="./data", help="trans output path")


args = parser.parse_args()


g2p = G2P()
remove_digits = str.maketrans('', '', string.digits)

unlabel_list = []
for spk in train_spk:
    paths = os.path.join(args.l2arctic_dir, spk, 'annotation/*.TextGrid')
    anno_list = [os.path.split(p)[1].split('.')[0] for p in glob(paths)]

    paths = os.path.join(args.l2arctic_dir, spk, 'transcript/*.txt')
    for p in glob(paths):
        id_ = os.path.split(p)[1].split('.')[0]
        if id_ not in anno_list:
            unlabel_list.append(p)

pathlib.Path(os.path.join(args.save_path, 'train_u')).mkdir(exist_ok=True)
with open(os.path.join(args.save_path, 'train_u', "trans_g2p"), 'w') as f:
    for spk in tqdm(train_spk):
        for trans_file in unlabel_list:
            with open(trans_file, 'r') as r:
                line = r.readline()
            phone_seq = ['sil']
            for word in line.split(' '):
                phone_seq += [x.lower().translate(remove_digits) for x in g2p.convert(word)]
                phone_seq += ['blank']
            phone_seq[-1] = 'sil'
            utt_id = '_'.join([spk, os.path.split(trans_file)[1].split('.')[0]])
            f.write(' '.join([utt_id, ' '.join(phone_seq)]) + "\n")
