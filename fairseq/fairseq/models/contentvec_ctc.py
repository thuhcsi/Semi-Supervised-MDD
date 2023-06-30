# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional, Dict, List, Tuple

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.models.contentvec import ContentvecConfig
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "group_norm_masked", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class ContentvecAsrConfig(ContentvecConfig):
    mask_channel_before: bool = False
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded contentvec args
    w2v_args: Any = None
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    conv_feature_layers: Optional[str] = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": (
                "string describing convolutional feature extraction "
                "layers in form of a python list that contains "
                "[(dim, kernel_size, stride), ...]"
            ),
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    target_layer: int = field(
        default=6, metadata={"help": "using the [target_layer] layer feature for training"}
    )


@dataclass
class ContentvecCtcConfig(ContentvecAsrConfig):
    blank_weight: float = 0
    blank_mode: str = "add"
    apply_mask: bool = True
    # mask_channel_prob: float = 0.5
    # mask_channel_length: int = 64
    # layerdrop: float = 0.05
    # activation_dropout


class ContentvecEncoder(FairseqEncoder):
    def __init__(self, cfg: ContentvecAsrConfig, output_size=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        # model = task.build_model(w2v_args.model)
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([cfg.w2v_path])
        model = models[0]

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        super().__init__(task.source_dictionary)

        self.target_layer = cfg.target_layer
        d = 768

        self.w2v_model = model

        self.remove_pretraining_modules()

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        targ_d = None
        self.proj = None

        if output_size is not None:
            targ_d = output_size
        elif getattr(cfg, "encoder_embed_dim", d) != d:
            targ_d = cfg.encoder_embed_dim

        if targ_d is not None:
            self.proj = Linear(d, targ_d)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def extract_features(self, source, padding_mask, **kwargs):
        features = self.w2v_model.forward_features(source, padding_mask)
        if padding_mask is not None:
            padding_mask = self.w2v_model.forward_padding_mask(features, padding_mask)
        features = features.transpose(1, 2)
        features = self.w2v_model.layer_norm(features)
        #unmasked_features = features.clone()

        if self.w2v_model.post_extract_proj is not None:
            features = self.w2v_model.post_extract_proj(features)

        features = self.w2v_model.dropout_input(features)
        #unmasked_features = self.dropout_features(unmasked_features)

        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.w2v_model.encoder(
            x,
            None,
            padding_mask=padding_mask,
            layer=self.target_layer-1,
            tap=False
        )
        return {
            'x': x, 
            'padding_mask': padding_mask,
            'layer_results': None
        }
    
    def remove_pretraining_modules(self):
        # self.w2v_model.post_extract_proj = None
        self.w2v_model.final_proj = None
        self.w2v_model.layer_proj = None
        # self.w2v_model.encoder = None

    def forward(self, source, padding_mask, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.extract_features(**w2v_args)

            x = res["x"]
            padding_mask = res["padding_mask"]
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": padding_mask,  # B x T,
            "layer_results": res["layer_results"],
        }

    def forward_torchscript(self, net_input):
        if torch.jit.is_scripting():
            return self.forward(net_input["source"], net_input["padding_mask"])
        else:
            return self.forward_non_torchscript(net_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
    

@register_model("contentvec_ctc", dataclass=ContentvecCtcConfig)
class ContentvecCtc(BaseFairseqModel):
    def __init__(self, cfg: ContentvecCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: ContentvecCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = ContentvecEncoder(cfg, len(task.target_dictionary))
        w2v_encoder.requires_grad_(True)
        return cls(cfg, w2v_encoder)

    def get_logits(self, net_output, normalize=False):
        logits = net_output["encoder_out"]
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"].T][..., 0] = float("inf")
            logits[net_output["padding_mask"].T][..., 1:] = float("-inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
