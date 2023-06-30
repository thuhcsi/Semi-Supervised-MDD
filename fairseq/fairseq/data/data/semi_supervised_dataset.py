import logging
import os
import sys
import logging
import io

import torch

from fairseq.data import FileAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressionLevel


logger = logging.getLogger(__name__)





class FileAudioDatasetWithTrans(FileAudioDataset):
    def __init__(
        self,
        manifest_path,
        trans_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        **mask_compute_kwargs,
    ):
        super().__init__(manifest_path,
        sample_rate,
        max_sample_size,
        min_sample_size,
        shuffle,
        pad,
        normalize,
        num_buckets,
        compute_mask_indices,
        text_compression_level,
        **mask_compute_kwargs,)

        self.trans = []
        with open(trans_path, 'r') as f:
            lines  = f.readlines()
            for line in lines:
                _, tran =  line.split(' ', 1)
                self.trans.append(tran.strip())

        def __getitem__(self, index):
            import soundfile as sf
            fn = self.fnames[index]
            fn = fn if isinstance(self.fnames, list) else fn.as_py()
            fn = self.text_compressor.decompress(fn)
            path_or_fp = os.path.join(self.root_dir, fn)
            _path, slice_ptr = parse_path(path_or_fp)
            if len(slice_ptr) == 2:
                byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
                assert is_sf_audio_data(byte_data)
                path_or_fp = io.BytesIO(byte_data)

            wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate)
            return {"id": index, "source": feats, 'tran': self.trans[index]}
