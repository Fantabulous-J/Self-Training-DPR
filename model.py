import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLayer
from transformers.utils import ModelOutput

from arguments import ModelArguments, DataArguments, BiEncoderTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class BiEncoderOutput(ModelOutput):
    query_vector: Tensor = None
    passage_vector: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


@dataclass
class SelfDistillBiEncoderOutput(BiEncoderOutput):
    query_vector: Tuple[Tensor, List[Tensor]] = None
    passage_vector: Tuple[Tensor, List[Tensor]] = None


@dataclass
class CrossEncoderOutput(ModelOutput):
    loss: Tensor = None
    logits: Tensor = None
    query_vector: Tensor = None
    passage_vector: Tensor = None


@dataclass
class PretrainCrossEncoderOutput(ModelOutput):
    loss: Tensor = None
    logits: Tensor = None
    mlm_loss: Tensor = None


class BiEncoder(nn.Module):
    def __init__(self,
                 query_encoder: PreTrainedModel,
                 passage_encoder: PreTrainedModel,
                 model_args: ModelArguments = None,
                 data_args: DataArguments = None,
                 train_args: BiEncoderTrainingArguments = None):
        super(BiEncoder, self).__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            assert dist.is_initialized() and dist.get_world_size() > 1, \
                ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        if self.train_args.recover_query:
            self.mlm_head = BertOnlyMLMHead(self.query_encoder.config)
            self.decoder = nn.ModuleList(
                [BertLayer(self.query_encoder.config) for _ in range(self.train_args.num_decoder_layer)]
            )

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.orig_params = tuple(self.parameters())
        self.orig_named_parameters = list(self.named_parameters())

    @classmethod
    def build(cls,
              model_args: ModelArguments,
              data_args: DataArguments,
              train_args: BiEncoderTrainingArguments,
              model_name_or_path: str = None,
              **hf_kwargs):
        if model_name_or_path is not None:
            model_args.model_name_or_path = model_name_or_path
        logger.info(f"Load model from {model_args.model_name_or_path}")
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.shared_encoder:
                query_encoder = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                passage_encoder = query_encoder
            else:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                query_encoder = AutoModel.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                passage_encoder = AutoModel.from_pretrained(_psg_model_path, **hf_kwargs)
        else:
            query_encoder = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            if model_args.shared_encoder:
                passage_encoder = query_encoder
            else:
                passage_encoder = deepcopy(query_encoder)

        model = cls(query_encoder=query_encoder,
                    passage_encoder=passage_encoder,
                    model_args=model_args,
                    data_args=data_args,
                    train_args=train_args)

        return model

    def forward(self,
                query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None,
                distillation: bool = False,
                teacher_scores: Tensor = None,
                compute_loss: bool = False,
                only_query: bool = False,
                only_passage: bool = False):

        query_vector = self.encode_query(query)
        if only_query:
            return BiEncoderOutput(query_vector=query_vector)
        passage_vector = self.encode_passage(passage)
        if only_passage:
            return BiEncoderOutput(passage_vector=passage_vector)

        if distillation:
            if compute_loss:
                batch_size = query_vector.size(0)
                passage_vector = torch.reshape(passage_vector, [batch_size, -1, passage_vector.size(-1)])
                student_scores = torch.sum(query_vector.unsqueeze(1) * passage_vector, dim=-1)

                teacher_scores = torch.reshape(teacher_scores, [batch_size, -1])
                teacher_scores = torch.softmax(teacher_scores * self.teacher_temp.exp(), dim=1)

                student_scores = torch.log_softmax(student_scores * self.student_temp.exp(), dim=1)
                loss = torch.nn.functional.kl_div(student_scores, teacher_scores, reduction='batchmean')
                return BiEncoderOutput(loss=loss)

            return BiEncoderOutput(
                query_vector=query_vector,
                passage_vector=passage_vector,
            )

        if self.training:
            if self.train_args.negatives_x_device:
                query_vector = self.dist_gather_tensor(query_vector)
                passage_vector = self.dist_gather_tensor(passage_vector)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device else self.train_args.per_device_train_batch_size

            scores = torch.matmul(query_vector, passage_vector.transpose(0, 1))

            # todo: try score scaling
            if self.train_args.retriever_score_scaling:
                scores = scores / math.sqrt(self.query_encoder.config.hidden_size)

            scores = scores.view(effective_bsz, -1)

            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            target = target * self.data_args.train_n_passages
            loss = self.cross_entropy(scores, target)
            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        else:
            loss = None
            scores = (query_vector * passage_vector).sum(1)

        return BiEncoderOutput(
            loss=loss,
            scores=scores,
            query_vector=query_vector,
            passage_vector=passage_vector,
        )

    @staticmethod
    def encode(model, inputs, pooling='cls'):
        if inputs is None:
            return None
        outputs = model(**inputs, return_dict=True)
        last_hidden_state = outputs.last_hidden_state

        if pooling == 'cls':
            return last_hidden_state[:, 0]
        else:
            assert pooling == 'mean', pooling
            attention_mask = inputs['attention_mask']
            last_hidden_state = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode_passage(self, passage):
        pooling = 'mean' if 'contriever' in self.model_args.model_name_or_path else 'cls'
        return self.encode(self.passage_encoder, passage, pooling)

    def encode_query(self, query):
        pooling = 'mean' if 'contriever' in self.model_args.model_name_or_path else 'cls'
        return self.encode(self.query_encoder, query, pooling)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        if self.model_args.shared_encoder:
            self.query_encoder.save_pretrained(output_dir)
        else:
            os.makedirs(os.path.join(output_dir, 'query_model'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'passage_model'), exist_ok=True)
            self.query_encoder.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.passage_encoder.save_pretrained(os.path.join(output_dir, 'passage_model'))