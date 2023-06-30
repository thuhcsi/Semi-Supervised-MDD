import sys
sys.path.append('./fairseq')
import os
import torch
import fairseq
import pathlib
import argparse
import numpy as np
import soundfile as sf

from tqdm import tqdm
from itertools import groupby


class Decoder:
    def __init__(self, json_dict):
        self.dict = json_dict
        self.look_up = np.asarray(list(self.dict.keys()))

    def decode(self, ids):
        converted_tokens = self.look_up[ids]
        fused_tokens = [tok[0] for tok in groupby(converted_tokens)]
        output = ' '.join([tok for tok in fused_tokens if tok != '<s>'])
        return output
    
parser = argparse.ArgumentParser(description="decode phoneme sequences")
parser.add_argument("--checkpoint_path", required=True, type=str)
parser.add_argument("--config_name", required=True, type=str)
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--segment", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)


def main(args):
    device = torch.device('cuda')

    with open(os.path.join(args.data_dir, args.segment, 'wav.scp'), 'r') as f:
        wav_files = f.readlines()

    model, cfg = fairseq.checkpoint_utils.load_model_ensemble(
    [args.checkpoint_path], arg_overrides={})
    model = model[0]
    model.eval()

    model.to(device=device)

    json_dict = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    count = 4
    with open(os.path.join(args.data_dir, 'dict.wrd.txt'), 'r') as f:
        for line in f.readlines():
            json_dict[line.split(' ')[0].strip()] = count
            count += 1

    decoder = Decoder(json_dict=json_dict)

    output_dir = os.path.join(args.output_dir, args.config_name)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    w1 = open(os.path.join(output_dir, 'decode_seq'), 'w+')

    for wav_file in tqdm(wav_files):
        wav_id, wav_path = wav_file.split(' ', 1)

        with torch.no_grad():
            input_sample = torch.tensor(sf.read(wav_path.strip())[0])[None, :].to(torch.float32).to(device=device)

            logits = model(source=input_sample, padding_mask=None)["encoder_out"]
            predicted_ids = torch.argmax(logits[:, 0], axis=-1).cpu()

        phn_seq = decoder.decode(predicted_ids)
        w1.write(wav_id + " " + phn_seq + "\n")
    w1.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

print("Prediction: ", )