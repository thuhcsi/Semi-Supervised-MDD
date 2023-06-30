# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.focal_ctc import FocalCtcCriterionConfig, FocalCtcCriterion
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class SemiSupervisedFocalCtcCriterionConfig(FocalCtcCriterionConfig):
    alpha: float = field(
        default=.5,
        metadata={"help": "weight of loss on unlabeled data"},
    )


@register_criterion("semi_supervised_focal_ctc", dataclass=SemiSupervisedFocalCtcCriterionConfig)
class SemiSupervisedCtcCriterion(FocalCtcCriterion):
    def __init__(self, cfg: SemiSupervisedFocalCtcCriterionConfig, task: FairseqTask):
        super().__init__(cfg, task)
        self.alpha = cfg.alpha

    def forward(self, model, sample, reduce=True):
        if not self.training:
            return super().forward(model, sample, reduce)
        
        alpha = 1.0
        batch_type = 'labeled'
        if 'target_u' in sample:
            alpha = self.alpha
            batch_type = 'unlabeled'
            sample['target'] = sample['target_u']

        loss, sample_size, logging_output = super().forward(model, sample, reduce)
        logging_output['batch_type'] = batch_type
        

        return torch.tensor(alpha, device=loss.device) * loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        if sample_size == 0:
            return
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )