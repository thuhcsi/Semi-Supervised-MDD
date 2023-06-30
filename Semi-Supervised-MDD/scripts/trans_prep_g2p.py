import os
import sys
import string
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from g2p.en_us import G2P


train_spk = ["MBMPS","ERMS","TLV","PNV","ASI","RRBI","TXHC","LXC","HJK","HKK","ABA","SKA"]
dev_spk = ["EBVS","THV","TNI","BWC","YDCK","YBAA"]
test_spk = ["NJS","HQTV","SVBI","NCC","YKWK","ZHAA"] 
load_error_file = ["YDCK/arctic_a0209.txt",
                  "YDCK/arctic_a0272.txt"]


parser = argparse.ArgumentParser(description="Prepare trans data")
parser.add_argument("--l2arctic_dir", default="/ceph/home/ywx20/data/L2_ARTIC_v5", help="l2-Arctic am output path")
parser.add_argument("--timit_dir", default="/ceph/home/ywx20/data/TIMIT", help="timit am output path")
parser.add_argument("--save_path", default="./data", help="trans output path")


args = parser.parse_args()


g2p = G2P()
remove_digits = str.maketrans('', '', string.digits)

map_dict = {0: 'train', 1: 'dev', 2: 'test'}
for i, spk_list in enumerate([train_spk, dev_spk, test_spk]):
    with open(os.path.join(args.save_path, map_dict[i], "trans_g2p"), 'w') as f:
        for spk in tqdm(spk_list):
            trans_file_list = os.listdir(os.path.join(args.l2arctic_dir, spk, "transcript"))
            for trans_file in trans_file_list:
                with open(os.path.join(args.l2arctic_dir, spk, "transcript", trans_file), 'r') as r:
                    line = r.readline()
                phone_seq = ['sil']
                for word in line.split(' '):
                    phone_seq += [x.lower().translate(remove_digits) for x in g2p.convert(word)]
                    phone_seq += ['blank']
                phone_seq[-1] = 'sil'
                utt_id = '_'.join([spk, trans_file.split('.')[0]])
                f.write(' '.join([utt_id, ' '.join(phone_seq)]) + "\n")

    if i > 0:
        continue
    # TIMIT
    trans_file_list = glob(args.timit_dir + '/*/DR*/*/*.TXT')
    with open(os.path.join(args.save_path, map_dict[i], "trans_g2p"), 'a') as f:
        for trans_file in tqdm(trans_file_list):
            with open(trans_file, 'r') as r:
                line = r.readline()
            line = line.split(' ', 2)[2].translate(str.maketrans('', '', string.punctuation)).strip('\n')
            phone_seq = ['sil']
            for word in line.split(' '):
                phone_seq += [x.lower().translate(remove_digits) for x in g2p.convert(word)]
                phone_seq += ['blank']
            phone_seq[-1] = 'sil'
            splited = trans_file.split('/')
            utt_id = '_'.join([splited[-2], splited[-1].split('.')[0]])
            f.write(' '.join([utt_id, ' '.join(phone_seq)]) + "\n")