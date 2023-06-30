# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import torch
import json
import editdistance

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from itertools import groupby
from collections import OrderedDict

from fairseq.data import data_utils
from fairseq.data import AddTargetDataset, Dictionary, encoders
from fairseq.data.semi_supervised_dataset import FileAudioDatasetWithTrans
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
from fairseq.optim.amp_optimizer import AMPOptimizer

from . import register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class AudioMeanTeacherConfig(AudioPretrainingConfig):
    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: Optional[str] = field(
        default=None, metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); "
                    "required if using --eval-bleu; use 'space' to disable "
                    "detokenization; see fairseq.data.encoders for other options"
        }
    )
    eval_bleu_detok_args: str = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False,
        metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None, metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={"help": "generation args for BLUE scoring, e.g., "
                          "'{\"beam\": 4, \"lenpen\": 0.6}'"}
    )
    eval_bleu_print_samples: bool = field(
        default=False,
        metadata={"help": "print sample generations during validation"}
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
    trans_path: str = field(default='./data/train_u/trans_g2p', metadata={'help': 'path to transcription phone sequences'})
    editdistance_threshold: float = field(default=0.7, metadata={'help': 'distance threshold for reject unlabeld samples'})    
    sample_reject: bool = field(
        default=False,
        metadata={"help": "whether to reject unlabeld samples"}
    )

@register_task("audio_mean_teacher", dataclass=AudioMeanTeacherConfig)
class AudioMeanTeacherTask(AudioPretrainingTask):
    """ """

    cfg: AudioMeanTeacherConfig

    def __init__(
        self,
        cfg: AudioMeanTeacherConfig,
    ):
        super().__init__(cfg)
        self.blank_symbol = "<s>"

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def load_dataset(self, split: str, task_cfg: AudioMeanTeacherConfig = None, **kwargs):
        super().load_dataset(split, task_cfg, **kwargs)

        task_cfg = task_cfg or self.cfg
        assert task_cfg.labels is not None
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )
        data_path = self.cfg.data
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f) if i not in skipped_indices
            ]

        assert len(labels) == len(self.datasets[split]), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(self.datasets[split])}) do not match"
        )

        process_label = LabelEncoder(self.target_dictionary)

        labeled_dataset = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                label_len_fn=label_len_fn,
                add_to_input=task_cfg.get("autoregressive", False),
                text_compression_level=text_compression_level
            )
        if split == 'train':
            data_path = self.cfg.data
            task_cfg = task_cfg or self.cfg

            # upgrade old task
            if isinstance(task_cfg, Namespace):
                if not hasattr(task_cfg, "autoregressive"):
                    task_cfg.autoregressive = not task_cfg.criterion == "ctc"

            text_compression_level = getattr(
                TextCompressionLevel, str(self.cfg.text_compression_level)
            )

            manifest_path = os.path.join(data_path, "{}.tsv".format('train_u'))

            self.datasets['train_u'] = FileAudioDatasetWithTrans(
                manifest_path=manifest_path,
                trans_path= self.cfg.trans_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                text_compression_level=text_compression_level,
                **self._get_mask_precompute_kwargs(task_cfg),
            )


            self.datasets[split] = MultiCorpusDataset(
                OrderedDict(
                    [('labeled_dataset', labeled_dataset),
                    ('unlabeled_dataset', self.datasets['train_u'])]
                ),
                distribution=[0.5, 0.5],
                seed=42,
                batch_sample=True
            )

        else:
            self.datasets[split] = labeled_dataset

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        p = next(model.ema.model.parameters())
        ema_device = p.device
        ema_dtype = p.dtype
        device = sample['net_input']['source'].device
        dtype = sample['net_input']['source'].dtype
        if 'target' not in sample:
            model.eval()
            if ema_device != device or ema_dtype != dtype:
                logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
                model.ema.model = model.ema.model.to(dtype=dtype, device=device)
                ema_dtype = dtype

                def to_device(d):
                    for k, p in d.items():
                        if isinstance(d[k], dict):
                            to_device(d[k])
                        else:
                            d[k] = p.to(device=device)

                to_device(model.ema.fp32_params)
            # pseudo labling
            with torch.no_grad():
                logits = model.ema.model(source=sample['net_input']['source'], padding_mask=None)["encoder_out"]
                predicted_ids = torch.argmax(logits, axis=-1).transpose(0,1).cpu().tolist()

                l = []
                selected = []
                for i, seq in enumerate(predicted_ids):
                    seq_tenor = torch.tensor([tok[0] for tok in groupby(seq) if tok[0] != 0])
                    hyp = ' '.join([x for x in [self.target_dictionary[x] for x in seq_tenor] if  x not in  ['<s>', 'sil']])
                    distance = editdistance.eval(
                        sample['tran'][i],
                        hyp
                    ) / len(sample['tran'][i].split(' '))
                    if self.cfg.sample_reject and distance > self.cfg.editdistance_threshold:
                        continue
                    selected.append(i)
                    l.append(seq_tenor)

                sample['net_input']['source'] = sample['net_input']['source'][selected]
                sample['net_input']['padding_mask'] = None

                if len(selected)>0:
                    pass
                    # print(f'selected {len(selected)} / {len(predicted_ids)}')
                    # print(sample['tran'][selected[0]])
                    # print(' '.join([x for x in [self.target_dictionary[x] for x in l[0]] if  x not in  ['<s>', 'sil']]))
                    # print(editdistance.eval(
                    #     sample['tran'][selected[0]],
                    #     ' '.join([x for x in [self.target_dictionary[x] for x in l[0]] if  x not in  ['<s>', 'sil']])
                    # )/len(sample['tran'][selected[0]].split(' ')))
                else:
                    return 0., 0, {}

                sample['target_u'] = data_utils.collate_tokens(l, pad_idx=self.target_dictionary.pad(), left_pad=False).to(device=device)

                model.train()
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        if self.cfg.eval_bleu and self.cfg.autoregressive:
            metrics = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = metrics.sys_len
            logging_output['_bleu_ref_len'] = metrics.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(metrics.counts) == 4
            for i in range(4):
                logging_output[f"_bleu_counts_{i}"] = metrics.counts[i]
                logging_output[f"_bleu_totals_{i}"] = metrics.totals[i]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        if self.cfg.eval_bleu and self.cfg.autoregressive:
            assert self.cfg.eval_bleu_detok is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )
            gen_args = json.loads(self.cfg.eval_bleu_args)
            gen_args = Namespace(**gen_args)
            self.sequence_generator = self.build_generator([model], gen_args)

        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, is_ref):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if is_ref else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens'], is_ref=False))
            refs.append(
                decode(
                    utils.strip_pad(
                        sample['target'][i],
                        self.target_dictionary.pad()
                    ),
                    is_ref=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info('H-{} {}'.format(sample["id"][0], hyps[0]))
            logger.info('T-{} {}'.format(sample["id"][0], refs[0]))

        eval_tokenization = 'none' if self.cfg.eval_tokenized_bleu else '13a'
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=eval_tokenization)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.cfg.eval_wer:
            zero = torch.scalar_tensor(0.0)
            num_char_errors = sum(
                log.get("_num_char_errors", zero) for log in logging_outputs
            )
            num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
            num_word_errors = sum(
                log.get("_num_word_errors", zero) for log in logging_outputs
            )
            num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
            metrics.log_scalar("_num_char_errors", num_char_errors)
            metrics.log_scalar("_num_chars", num_chars)
            metrics.log_scalar("_num_word_errors", num_word_errors)
            metrics.log_scalar("_num_words", num_words)
            if num_chars > 0:
                metrics.log_derived(
                    "uer",
                    lambda meters: meters["_num_char_errors"].sum
                    * 100.0
                    / meters["_num_chars"].sum
                    if meters["_num_chars"].sum > 0
                    else float("nan"),
                )
            if num_words > 0:
                metrics.log_derived(
                    "wer",
                    lambda meters: meters["_num_word_errors"].sum
                    * 100.0
                    / meters["_num_words"].sum
                    if meters["_num_words"].sum > 0
                    else float("nan"),
                )
        if self.cfg.eval_bleu:
            len_keys = ["_bleu_sys_len", "_bleu_ref_len"]
            count_keys = [f"_bleu_counts_{i}" for i in range(4)]
            total_keys = [f"_bleu_totals_{i}" for i in range(4)]
            for k in len_keys + count_keys + total_keys:
                metrics.log_scalar(
                    k, sum(log.get(k, 0) for log in logging_outputs)
                )

            import sacrebleu
            metrics.log_derived(
                'bleu',
                lambda meters: sacrebleu.compute_bleu(
                    correct=[meters[k].sum for k in count_keys],
                    total=[meters[k].sum for k in total_keys],
                    sys_len=meters['_bleu_sys_len'].sum,
                    ref_len=meters['_bleu_ref_len'].sum,
                    smooth_method="exp"
                ).score
            )
