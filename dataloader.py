import copy
import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, DataCollatorForWholeWordMask

from arguments import DataArguments
from utils import noise_strategy

logger = logging.getLogger(__name__)


class GenericDataLoader:

    def __init__(self, data_folder: str = None, corpus_file: str = "corpus.tsv", query_file: str = "train.query.txt",
                 qrels_file: str = "qrels.train.tsv"):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_file = os.path.join(data_folder, qrels_file) if data_folder else qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
            if 'wikipedia' in self.qrels_file or 'pretrain' in self.qrels_file:
                logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
                logger.info("Query Example: %s", list(self.queries.values())[0])

        if os.path.exists(self.qrels_file):
            self._load_qrels(split=split)
            if 'wikipedia' not in self.qrels_file and 'pretrain' not in self.qrels_file:
                self.queries = {qid: self.queries[qid] for qid in self.qrels if qid in self.queries}
                logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
                logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def load_queries(self) -> Dict[str, str]:
        if not len(self.queries):
            logger.info("Loading Queries...")
        self._load_queries()
        logger.info("Loaded %d Queries.", len(self.queries))
        logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.queries

    def _load_corpus(self):
        if self.corpus_file.endswith(".tsv"):
            with open(self.corpus_file, encoding='utf-8') as fIn:
                reader = csv.reader(fIn, delimiter="\t")
                for row in tqdm(reader):
                    if not row[0] == "id":
                        self.corpus[row[0]] = {
                            "title": row[2],
                            "text": row[1]
                        }
        else:
            with open(self.corpus_file, encoding='utf-8') as fIn:
                for line in tqdm(fIn.readlines()):
                    corpus_id, title, text = line.strip().split('\t')
                    self.corpus[corpus_id] = {
                        'text': text,
                        'title': "" if title == '-' else title,
                    }

    def _load_queries(self):
        if self.query_file.endswith('tsv') or  self.query_file.endswith('csv'):
            with open( self.query_file, encoding='utf-8') as fIn:
                reader = csv.reader(fIn, delimiter="\t")
                for row in tqdm(reader):
                    if not row[0] == "qid":
                        self.queries[str(len(self.queries))] = row[0]
        else:
            with open(self.query_file, encoding='utf-8') as fIn:
                for line in fIn:
                    try:
                        query_id, text = line.strip().split('\t')
                        self.queries[query_id] = text
                    except:
                        logger.info(line)
                        exit(-1)

    def _load_qrels(self, split):
        with open(self.qrels_file, encoding='utf-8') as fIn:
            reader = csv.reader(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            if 'wikipedia' in self.qrels_file:
                if split != 'test':
                    for idx, [query_id, _, corpus_id, _] in enumerate(reader):
                        if query_id not in self.qrels:
                            self.qrels[query_id] = [corpus_id]
                        else:
                            self.qrels[query_id].append(corpus_id)
                else:
                    for idx, [query_id, corpus_id] in enumerate(reader):
                        if query_id not in self.qrels:
                            self.qrels[query_id] = [corpus_id]
                        else:
                            self.qrels[query_id].append(corpus_id)
            else:
                if split == 'train':
                    for idx, [query_id, _, corpus_id, score] in enumerate(reader):
                        if query_id not in self.qrels:
                            self.qrels[query_id] = {corpus_id: score}
                        else:
                            self.qrels[query_id][corpus_id] = score
                else:
                    for idx, [query_id, corpus_id] in enumerate(reader):
                        if query_id not in self.qrels:
                            self.qrels[query_id] = {corpus_id: 1}
                        else:
                            self.qrels[query_id][corpus_id] = 1


class BiEncoderDataset(Dataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 instruction: str = '',
                 trainer=None):
        super(BiEncoderDataset, self).__init__()

        self.corpus = corpus
        self.queries = queries
        self.data_path = data_path

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.trainer = trainer

        self.instruction = instruction

        # negative_size = self.data_args.train_n_passages - 1
        self.qrels = []
        if isinstance(data_path, List) and isinstance(data_path[0], Dict):
            self.qrels = data_path
        else:
            with open(data_path, 'r') as f:
                for jsonline in tqdm(f.readlines()):
                    if 'qgen' in data_path or 'pretrain' in data_path:
                        # neg_pids = example['neg']
                        # example['neg'] = random.sample(neg_pids, negative_size)
                        self.qrels.append(jsonline)
                    else:
                        example = json.loads(jsonline)
                        if len(example['pos']) == 0 or len(example['neg']) == 0:
                            continue
                        self.qrels.append(example)
                # if 'qgen' in data_path and 'wikipedia' in data_path:
                #     random.shuffle(self.qrels)
                #     self.qrels = self.qrels[:2000000]

            logger.info("Loaded %d Queries from %s.", len(self.qrels), data_path)

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        example = self.qrels[idx]
        if isinstance(example, str):
            example = json.loads(example)
        _hashed_seed = hash(idx + self.trainer.args.seed)
        epoch = int(self.trainer.state.epoch if hasattr(self.trainer, 'state') else self.trainer.epoch)

        qid = example['qid']
        if self.instruction != '':
            encoded_query = self.encode(self.instruction + self.queries[qid], max_length=self.data_args.max_query_length)
        else:
            encoded_query = self.encode(self.queries[qid], max_length=self.data_args.max_query_length)

        encoded_passages = []

        pos_pids = example['pos']
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = pos_pids[(_hashed_seed + epoch) % len(pos_pids)]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            _offset = epoch * negative_size % len(neg_pids)
            negs = [x for x in neg_pids]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        # pos_pids = example['pos']
        # pos_pid = random.sample(pos_pids, 1)[0]
        #
        # neg_pids = example['neg'][:30]
        # negative_size = self.data_args.train_n_passages - 1
        # if len(neg_pids) < negative_size:
        #     negs = random.choices(neg_pids, k=negative_size)
        # elif self.data_args.train_n_passages == 1:
        #     negs = []
        # else:
        #     negs = random.sample(neg_pids, negative_size)

        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            title = passage['title']
            text = passage['text']
            passage = title + self.tokenizer.sep_token + text
            encoded_passage = self.encode(passage, max_length=self.data_args.max_passage_length)
            encoded_passages.append(encoded_passage)

        return encoded_query, encoded_passages

    def encode(self, text, max_length):
        return self.tokenizer.encode_plus(
            text,
            max_length=max_length,
            truncation='only_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )


class NoisyBiEncoderDataset(BiEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 noise_type: str,
                 noise_prob: float,
                 ):
        super(NoisyBiEncoderDataset, self).__init__(corpus=corpus,
                                                    queries=queries,
                                                    data_path=data_path,
                                                    tokenizer=tokenizer,
                                                    data_args=data_args)
        assert noise_type in noise_strategy.keys(), noise_type
        self.noise_strategy = noise_strategy[noise_type]
        self.noise_prob = noise_prob

    def __getitem__(self, idx):
        encoded_query, encoded_passages = super(NoisyBiEncoderDataset, self).__getitem__(idx)

        encoded_noisy_query = copy.deepcopy(encoded_query)
        input_ids = self.noise_strategy(encoded_noisy_query['input_ids'][1:-1], p=self.noise_prob)
        encoded_noisy_query['input_ids'] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

        encoded_noisy_passages = copy.deepcopy(encoded_passages)
        for passage in encoded_noisy_passages:
            input_ids = self.noise_strategy(passage['input_ids'][1:-1], p=self.noise_prob)
            passage['input_ids'] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

        return encoded_query, encoded_passages, encoded_noisy_query, encoded_noisy_passages


class CrossEncoderDataset(BiEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments
                 ):
        super(CrossEncoderDataset, self).__init__(corpus=corpus,
                                                  queries=queries,
                                                  data_path=data_path,
                                                  tokenizer=tokenizer,
                                                  data_args=data_args)

    def __getitem__(self, idx):
        example = self.qrels[idx]

        qid = example['qid']
        query = self.queries[qid]

        encoded_query_passages = []
        pos_pids = example['pos']
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = random.sample(pos_pids, 1)[0]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            negs = random.sample(neg_pids, negative_size)

        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            title = passage['title']
            text = passage['text']
            passage = title + self.tokenizer.sep_token + text
            encoded_query_passages.append(self.encode(query, passage))

        return encoded_query_passages

    def encode(self, query_text, passage_text):
        return self.tokenizer.encode_plus(
            query_text,
            passage_text,
            max_length=self.data_args.max_query_passage_length,
            truncation='longest_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )


class NoisyCrossEncoderDataset(BiEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 noise_type: str,
                 noise_prob: float,
                 ):
        super(NoisyCrossEncoderDataset, self).__init__(corpus=corpus,
                                                       queries=queries,
                                                       data_path=data_path,
                                                       tokenizer=tokenizer,
                                                       data_args=data_args)
        assert noise_type in noise_strategy.keys(), noise_type
        self.noise_strategy = noise_strategy[noise_type]
        self.noise_prob = noise_prob

    def encode_noisy_query_passage(self, qid, pos_pid, negs):
        query = self.queries[qid]
        encoded_query = self.encode(query, max_length=self.data_args.max_query_length)

        encoded_passages = []
        encoded_query_passages = []
        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            title = passage['title']
            text = passage['text']
            passage = title + self.tokenizer.sep_token + text
            encoded_passage = self.encode(passage, max_length=self.data_args.max_passage_length)
            encoded_passages.append(encoded_passage)
            encoded_query_passages.append(
                self.tokenizer.encode_plus(
                    query,
                    passage,
                    max_length=self.data_args.max_query_passage_length,
                    truncation='longest_first',
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=True,
                )
            )

        encoded_noisy_query = copy.deepcopy(encoded_query)
        noisy_query_ids = self.noise_strategy(encoded_noisy_query['input_ids'][1:-1], p=self.noise_prob)

        encoded_noisy_passages = copy.deepcopy(encoded_passages)
        noisy_passage_ids = []
        for passage in encoded_noisy_passages:
            input_ids = self.noise_strategy(passage['input_ids'][1:-1], p=self.noise_prob)
            noisy_passage_ids.append(input_ids)

        encoded_noisy_query_passages = []
        for i, passage in enumerate(noisy_passage_ids):
            if len(noisy_query_ids) == 0 or len(passage) == 0:
                encoded_noisy_query_passages.append(copy.deepcopy(encoded_query_passages[i]))
                continue
            encoded_noisy_query_passage = self.tokenizer.prepare_for_model(
                noisy_query_ids,
                passage,
                max_length=self.data_args.max_query_passage_length,
                truncation='longest_first',
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=True,
                prepend_batch_axis=True
            )
            if len(encoded_noisy_query_passage['input_ids']) > self.data_args.max_query_passage_length:
                logger.info('overflow!!!')
                logger.info(self.tokenizer.decode(noisy_query_ids))
                logger.info(self.tokenizer.decode(passage))
                encoded_noisy_query_passage['input_ids'] = \
                    encoded_noisy_query_passage['input_ids'][:(self.data_args.max_query_passage_length - 1)] + \
                    [self.tokenizer.sep_token_id]
                assert len(encoded_noisy_query_passage['input_ids']) == self.data_args.max_query_passage_length, \
                    len(encoded_noisy_query_passage['input_ids'])
                encoded_noisy_query_passage['token_type_ids'] = encoded_noisy_query_passage['token_type_ids'][
                                                                :self.data_args.max_query_passage_length]
                assert len(encoded_noisy_query_passage['token_type_ids']) == self.data_args.max_query_passage_length, \
                    len(encoded_noisy_query_passage['token_type_ids'])

            encoded_noisy_query_passages.append(encoded_noisy_query_passage)

        return encoded_query_passages, encoded_noisy_query_passages

    def __getitem__(self, idx):
        example = self.qrels[idx]
        if isinstance(example, str):
            example = json.loads(example)

        qid = example['qid']

        pos_pids = example['pos']
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = random.sample(pos_pids, 1)[0]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            negs = random.sample(neg_pids, negative_size)

        encoded_query_passages, encoded_noisy_query_passages = self.encode_noisy_query_passage(qid, pos_pid, negs)

        return encoded_query_passages, encoded_noisy_query_passages


class NoisyDistillDataset(NoisyCrossEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 teacher_tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 noise_type: str,
                 noise_prob: float,
                 ):
        super(NoisyDistillDataset, self).__init__(corpus=corpus,
                                                  queries=queries,
                                                  data_path=data_path,
                                                  tokenizer=tokenizer,
                                                  data_args=data_args,
                                                  noise_type=noise_type,
                                                  noise_prob=noise_prob)
        self.teacher_tokenizer = teacher_tokenizer

    def __getitem__(self, idx):
        # from BE to CE for unsupervised learning in BEIR
        example = self.qrels[idx]
        if isinstance(example, str):
            example = json.loads(example)

        qid = example['qid']

        pos_pids = example['pos']
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = random.sample(pos_pids, 1)[0]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            negs = random.sample(neg_pids, negative_size)

        _, encoded_noisy_query_passages = self.encode_noisy_query_passage(qid, pos_pid, negs)

        encoded_query = self.teacher_tokenizer.encode_plus(
            self.queries[qid],
            max_length=self.data_args.max_teacher_query_length,
            truncation='only_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        encoded_passages = []
        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            title = passage['title']
            text = passage['text']
            passage = title + self.teacher_tokenizer.sep_token + text
            encoded_passage = self.teacher_tokenizer.encode_plus(
                passage,
                max_length=self.data_args.max_teacher_passage_length,
                truncation='only_first',
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            encoded_passages.append(encoded_passage)

        return encoded_query, encoded_passages, encoded_noisy_query_passages


class PretrainCrossEncoderDataset(CrossEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments
                 ):
        super(PretrainCrossEncoderDataset, self).__init__(corpus=corpus,
                                                          queries=queries,
                                                          data_path=data_path,
                                                          tokenizer=tokenizer,
                                                          data_args=data_args)

    def __getitem__(self, idx):
        example = self.qrels[idx]

        qid = example['qid']
        query = self.corpus[qid]['text']

        encoded_query_passages = []
        pos_pids = example['pos']
        pos_pid = random.sample(pos_pids, 1)[0]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            negs = random.sample(neg_pids, negative_size)

        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            text = passage['text']
            encoded_query_passages.append(self.encode(query, text))

        return encoded_query_passages


class DistillDataset(BiEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 teacher_tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments
                 ):
        super(DistillDataset, self).__init__(corpus=corpus,
                                             queries=queries,
                                             data_path=data_path,
                                             tokenizer=tokenizer,
                                             data_args=data_args)
        self.teacher_tokenizer = teacher_tokenizer

    def __getitem__(self, idx):
        example = self.qrels[idx]
        if isinstance(example, str):
            example = json.loads(example)

        qid = example['qid']
        query = self.queries[qid]
        encoded_student_query = self.encode_student(query, self.data_args.max_query_length)

        encoded_student_passages = []
        encoded_query_passages = []
        pos_pids = example['pos']
        # reminder: current kd on NQ shuffled positives (it matters on gold-labelled data but not on synthetic data as
        # we always have only one positive)
        # todo: latter should try no_shuffle_positive for kd on gold data for NQ
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = random.sample(pos_pids, 1)[0]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        # # todo: for wikipedia corpus, use the top-negative_size negatives directly rather than random sampling
        # elif 'qgen' in self.data_path and 'wikipedia' in self.data_path:
        #     negs = neg_pids[:negative_size]
        else:
            negs = random.sample(neg_pids, negative_size)

        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            title = passage['title']
            text = passage['text']
            passage = title + self.tokenizer.sep_token + text
            encoded_student_passages.append(self.encode_student(passage, self.data_args.max_passage_length))
            encoded_query_passages.append(self.encode_teacher(query, passage))

        return encoded_student_query, encoded_student_passages, encoded_query_passages

    def encode_student(self, text, max_length):
        return self.tokenizer.encode_plus(
            text,
            max_length=max_length,
            truncation='only_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def encode_teacher(self, query_text, passage_text):
        return self.teacher_tokenizer.encode_plus(
            query_text,
            passage_text,
            max_length=self.data_args.max_query_passage_length,
            truncation='longest_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )


class NoisyLMDistillDataset(BiEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 teacher_tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 prompt: str,
                 noise_type: str,
                 noise_prob: float
                 ):
        super(NoisyLMDistillDataset, self).__init__(corpus=corpus,
                                               queries=queries,
                                               data_path=data_path,
                                               tokenizer=tokenizer,
                                               data_args=data_args)
        self.teacher_tokenizer = teacher_tokenizer
        self.prompt = prompt

        assert noise_type in noise_strategy.keys(), noise_type
        self.noise_strategy = noise_strategy[noise_type]
        self.noise_prob = noise_prob

    def __getitem__(self, idx):
        example = self.qrels[idx]
        if isinstance(example, str):
            example = json.loads(example)

        qid = example['qid']
        query = self.queries[qid]
        encoded_student_query = self.encode_student(query, self.data_args.max_query_length)

        encoded_student_passages = []
        pos_pids = example['pos']
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = random.sample(pos_pids, 1)[0]

        neg_pids = example['neg']
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            negs = random.sample(neg_pids, negative_size)

        queries = []
        passages = []
        for pid in [pos_pid] + negs:
            passage = self.corpus[pid]
            title = passage['title']
            text = passage['text']
            passage = title + self.tokenizer.sep_token + text
            encoded_student_passages.append(self.encode_student(passage, self.data_args.max_passage_length))
            queries.append(query)
            passages.append(self.prompt.format((title + " " + text).strip()))

        encoded_noisy_student_query = copy.deepcopy(encoded_student_query)
        input_ids = self.noise_strategy(encoded_noisy_student_query['input_ids'][1:-1], p=self.noise_prob)
        encoded_noisy_student_query['input_ids'] = \
            [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

        encoded_noisy_student_passages = copy.deepcopy(encoded_student_passages)
        for passage in encoded_noisy_student_passages:
            input_ids = self.noise_strategy(passage['input_ids'][1:-1], p=self.noise_prob)
            passage['input_ids'] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

        return encoded_noisy_student_query, encoded_noisy_student_passages, queries, passages

    def encode_student(self, text, max_length):
        return self.tokenizer.encode_plus(
            text,
            max_length=max_length,
            truncation='only_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )


class OnTheFlyDistillDataset(DistillDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 teacher_tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments
                 ):
        super(OnTheFlyDistillDataset, self).__init__(corpus=corpus,
                                                     queries=queries,
                                                     data_path=data_path,
                                                     tokenizer=tokenizer,
                                                     teacher_tokenizer=teacher_tokenizer,
                                                     data_args=data_args)

    def __getitem__(self, idx):
        example = self.qrels[idx]

        qid = example['qid']
        query = self.queries[qid]
        encoded_student_query = self.encode_student(query, self.data_args.max_query_length)

        return qid, encoded_student_query


class QueryGenDataset(Dataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 data_path: Union[str, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments):
        super(QueryGenDataset, self).__init__()
        self.corpus = corpus
        self.queries = queries

        self.qrels = []
        with open(data_path, 'r') as f:
            for jsonline in f.readlines():
                example = json.loads(jsonline)
                if len(example['pos']) == 0:
                    continue
                self.qrels.append((example['qid'], example['pos']))
        logger.info("Loaded %d Queries from %s.", len(self.qrels), data_path)

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        qid, pids = self.qrels[idx]

        query = self.queries[qid]
        if self.data_args.no_shuffle_positive:
            pid = pids[0]
        else:
            pid = random.sample(pids, 1)[0]

        passage = self.corpus[pid]
        text = passage['title'] + " " + passage['text']

        return query, text


class EncodeDataset(Dataset):
    def __init__(self,
                 data: Dict,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int,
                 is_query: bool,
                 start: int = -1,
                 end: int = -1):
        super(EncodeDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_query = is_query

        self.data = []
        for idx, text in data.items():
            self.data.append((idx, text))
        if start >=0 and end >= 0:
            self.data = self.data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_id, text = self.data[idx]
        if not self.is_query:
            text = text['title'] + self.tokenizer.sep_token + text['text']

        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )

        # if self.is_query:
        #     input_ids = noise_strategy['shuffle'](copy.deepcopy(encoded_text['input_ids'][1:-1]), p=0.2)
        #     encoded_text['input_ids'] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return text_id, encoded_text


class RerankingDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int):
        super(RerankingDataset, self).__init__()

        self.tokenizer = tokenizer
        self.corpus = corpus
        self.queries = queries
        self.max_length = max_length

        self.data = []
        with open(data_path) as f:
            for line in f.readlines():
                qid, pid, _ = line.strip().split()
                self.data.append((qid, pid))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, pid = self.data[idx]

        query = self.queries[qid]
        passage = self.corpus[pid]
        title = passage['title']
        text = passage['text']
        passage = title + self.tokenizer.sep_token + text

        encoded_query_passage = self.encode(query, passage)

        return qid, pid, encoded_query_passage

    def encode(self, query_text, passage_text):
        return self.tokenizer.encode_plus(
            query_text,
            passage_text,
            max_length=self.max_length,
            truncation='longest_first',
            padding=False,
            return_token_type_ids=True,
        )


@dataclass
class BiEncoderCollator(DataCollatorWithPadding):
    max_query_length: int = 32
    max_passage_length: int = 128

    def _encode(self, features):
        batch_query = [f[0] for f in features]
        batch_passage = [f[1] for f in features]

        if isinstance(batch_query[0], list):
            batch_query = sum(batch_query, [])
        if isinstance(batch_passage[0], list):
            batch_passage = sum(batch_passage, [])

        batch_query = self.tokenizer.pad(
            batch_query,
            padding='max_length',
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        batch_passage = self.tokenizer.pad(
            batch_passage,
            padding='max_length',
            max_length=self.max_passage_length,
            return_tensors="pt",
        )

        return batch_query, batch_passage

    def __call__(self, features):
        return self._encode(features)


@dataclass
class NoisyBiEncoderCollator(BiEncoderCollator):
    def __call__(self, features):
        batch_query, batch_passage = self._encode(features)

        batch_noisy_query = [f[2] for f in features]
        if isinstance(batch_noisy_query[0], list):
            batch_noisy_query = sum(batch_noisy_query, [])
        batch_noisy_query = self.tokenizer.pad(
            batch_noisy_query,
            padding='max_length',
            max_length=self.max_query_length,
            return_tensors="pt",
        )

        batch_noisy_passage = [f[3] for f in features]
        if isinstance(batch_noisy_passage[0], list):
            batch_noisy_passage = sum(batch_noisy_passage, [])
        batch_noisy_passage = self.tokenizer.pad(
            batch_noisy_passage,
            padding='max_length',
            max_length=self.max_passage_length,
            return_tensors="pt",
        )

        return batch_query, batch_passage, batch_noisy_query, batch_noisy_passage


@dataclass
class CrossEncoderCollator(DataCollatorWithPadding):
    max_query_passage_length: int = 156

    def __call__(self, features):
        batch_query_passage = [f for f in features]

        batch_query_passage = sum(batch_query_passage, [])

        batch_query_passage = self.tokenizer.pad(
            batch_query_passage,
            padding='max_length',
            max_length=self.max_query_passage_length,
            return_tensors='pt'
        )

        return batch_query_passage


@dataclass
class NoisyCrossEncoderCollator(CrossEncoderCollator):
    def __call__(self, features):
        batch_query_passage = [f[0] for f in features]
        batch_noisy_query_passage = [f[1] for f in features]

        batch_query_passage = sum(batch_query_passage, [])
        batch_noisy_query_passage = sum(batch_noisy_query_passage, [])

        batch_query_passage = self.tokenizer.pad(
            batch_query_passage,
            padding='max_length',
            max_length=self.max_query_passage_length,
            return_tensors='pt'
        )

        # try:
        #     batch_noisy_query_passage = self.tokenizer.pad(
        #         batch_noisy_query_passage,
        #         padding='max_length',
        #         max_length=self.max_query_passage_length,
        #         return_tensors='pt'
        #     )
        # except:
        #     for qp in batch_noisy_query_passage:
        #         if len(qp['input_ids']) > self.max_query_passage_length:
        #             print(self.tokenizer.decode(qp['input_ids']), qp['input_ids'], len(qp['input_ids']))
        #             print(qp['token_type_ids'])
        #             print('\n')
        #     exit(-1)

        batch_noisy_query_passage = self.tokenizer.pad(
            batch_noisy_query_passage,
            padding='max_length',
            max_length=self.max_query_passage_length,
            return_tensors='pt'
        )

        return batch_query_passage, batch_noisy_query_passage


@dataclass
class PretrainCrossEncoderCollator(DataCollatorForWholeWordMask):
    max_query_passage_length: int = 256

    def __post_init__(self):
        super(PretrainCrossEncoderCollator, self).__post_init__()
        self.specials = self.tokenizer.all_special_tokens

    def _whole_word_cand_indexes(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=256):
        cand_indexes = self._whole_word_cand_indexes(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _pad(self, seq, val=0):
        tgt_len = self.max_query_passage_length
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]

    def _encode_seq(self, sequences):
        encoded_examples = []
        masks = []
        token_type_ids = []
        mlm_masks = []

        for seq in sequences:
            tokens = [self.tokenizer._convert_id_to_token(token_id) for token_id in seq['input_ids']]
            mlm_mask = self._whole_word_mask(tokens)
            mlm_mask = self._pad(mlm_mask)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.pad(
                seq,
                padding='max_length',
                max_length=self.max_query_passage_length,
                return_tensors='pt'
            )

            masks.append(encoded['attention_mask'])
            token_type_ids.append(encoded['token_type_ids'] if 'token_type_ids' in encoded else None)
            encoded_examples.append(encoded['input_ids'])

            assert len(mlm_mask) == len(encoded['input_ids']), (len(mlm_mask), len(encoded['input_ids']))
            assert len(mlm_mask) == len(encoded['attention_mask']), (len(mlm_mask), len(encoded['attention_mask']))
            if 'token_type_ids' in encoded:
                assert len(mlm_mask) == len(encoded['token_type_ids']), (len(mlm_mask), len(encoded['token_type_ids']))

        inputs, labels = self.torch_mask_tokens(
            torch.stack(encoded_examples),
            torch.tensor(mlm_masks, dtype=torch.long)
        )

        return inputs, masks, token_type_ids, labels

    def __call__(self, features):
        batch_query_passage = sum(features, [])

        inputs, masks, token_type_ids, labels = self._encode_seq(batch_query_passage)

        batch_query_passage = {
            "input_ids": inputs,
            "attention_mask": torch.stack(masks),
            "token_type_ids": torch.stack(token_type_ids)
        }

        return batch_query_passage, labels


@dataclass
class PretrainBiEncoderCollator(BiEncoderCollator, PretrainCrossEncoderCollator):
    def __post_init__(self):
        super(PretrainBiEncoderCollator, self).__post_init__()
        self.max_query_passage_length = self.max_query_length

    def __call__(self, features):
        batch_query, batch_passage = self._encode(features)
        query_inputs, query_masks, _, query_labels = self._encode_seq([f[0] for f in features])
        query_mlm_inputs = {
            "input_ids": query_inputs,
            "attention_mask": torch.stack(query_masks),
        }

        return batch_query, batch_passage, query_mlm_inputs, query_labels


@dataclass
class PretrainCrossEncoderEncDecCollator(PretrainCrossEncoderCollator):
    enc_mlm_probability: float = 0.15
    dec_mlm_probability: float = 0.15

    def __call__(self, features):
        batch_query_passage = sum(features, [])

        # encoder masking
        self.mlm_probability = self.enc_mlm_probability
        enc_inputs, enc_masks, enc_token_type_ids, enc_labels = self._encode_seq(batch_query_passage)
        batch_encoder_query_passage = {
            "input_ids": enc_inputs,
            "attention_mask": torch.stack(enc_masks),
            "token_type_ids": torch.stack(enc_token_type_ids)
        }

        # decoder masking
        self.mlm_probability = self.dec_mlm_probability
        dec_inputs, dec_masks, dec_token_type_ids, dec_labels = self._encode_seq(batch_query_passage)
        batch_decoder_query_passage = {
            "input_ids": dec_inputs,
            "attention_mask": torch.stack(dec_masks),
            "token_type_ids": torch.stack(dec_token_type_ids)
        }

        return batch_encoder_query_passage, enc_labels, batch_decoder_query_passage, dec_labels


@dataclass
class DistillCollator(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizer = None
    teacher_tokenizer: PreTrainedTokenizer = None

    max_query_length: int = 32
    max_passage_length: int = 128
    max_query_passage_length: int = 156

    def __call__(self, features):
        batch_student_query = [f[0] for f in features]
        batch_student_passage = [f[1] for f in features]
        batch_query_passage = [f[2] for f in features]

        if isinstance(batch_student_query[0], list):
            batch_student_query = sum(batch_student_query, [])
        if isinstance(batch_student_passage[0], list):
            batch_student_passage = sum(batch_student_passage, [])
        if isinstance(batch_query_passage[0], list):
            batch_query_passage = sum(batch_query_passage, [])

        batch_student_query = self.tokenizer.pad(
            batch_student_query,
            padding='max_length',
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        batch_student_passage = self.tokenizer.pad(
            batch_student_passage,
            padding='max_length',
            max_length=self.max_passage_length,
            return_tensors="pt",
        )

        batch_query_passage = self.teacher_tokenizer.pad(
            batch_query_passage,
            padding='max_length',
            max_length=self.max_query_passage_length,
            return_tensors='pt'
        )

        return batch_student_query, batch_student_passage, batch_query_passage


@dataclass
class OnTheFlyDistillCollator(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizer = None
    max_query_length: int = 32

    def __call__(self, features):
        batch_query_ids = [f[0] for f in features]
        batch_student_query = [f[1] for f in features]

        if isinstance(batch_student_query[0], list):
            batch_student_query = sum(batch_student_query, [])

        batch_student_query = self.tokenizer.pad(
            batch_student_query,
            padding='max_length',
            max_length=self.max_query_length,
            return_tensors="pt",
        )

        return batch_query_ids, batch_student_query


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch_text_id = [x[0] for x in features]
        batch_text = [x[1] for x in features]
        batch_text = super().__call__(batch_text)

        return batch_text_id, batch_text


@dataclass
class RerankingCollator(DataCollatorWithPadding):
    max_query_passage_length: int = 156

    def __call__(self, features):
        batch_query_ids = [f[0] for f in features]
        batch_passage_ids = [f[1] for f in features]
        batch_query_passage = [f[2] for f in features]
        batch_query_passage = super().__call__(batch_query_passage)

        return batch_query_ids, batch_passage_ids, batch_query_passage


class QueryGenCollator(object):
    def __init__(self, tokenizer, text_maxlength=512, query_maxlength=64):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.query_maxlength = query_maxlength

    def __call__(self, batch):
        assert (batch[0][0] is not None)

        query = [ex[0] for ex in batch]
        query = self.tokenizer.batch_encode_plus(
            query,
            max_length=self.query_maxlength,
            padding='longest',
            return_tensors='pt',
            truncation=True,
        )
        query_ids = query["input_ids"]
        query_mask = query["attention_mask"].bool()
        query_ids = query_ids.masked_fill(~query_mask, -100)

        texts = [ex[1] for ex in batch]
        batch_texts = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.text_maxlength,
            padding='longest',
            return_tensors='pt',
            truncation=True
        )
        batch_texts['labels'] = query_ids

        return batch_texts


@dataclass
class NoisyLMDistillCollator(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizer = None
    teacher_tokenizer: PreTrainedTokenizer = None

    max_query_length: int = 64
    max_passage_length: int = 256
    max_teacher_query_length: int = 512
    max_teacher_passage_length: int = 512

    def __call__(self, features):
        batch_student_query = [f[0] for f in features]
        batch_student_passage = [f[1] for f in features]
        batch_queries = [f[2] for f in features]
        batch_passages = [f[3] for f in features]

        # student inputs
        if isinstance(batch_student_query[0], list):
            batch_student_query = sum(batch_student_query, [])
        if isinstance(batch_student_passage[0], list):
            batch_student_passage = sum(batch_student_passage, [])
        if isinstance(batch_queries[0], list):
            batch_queries = sum(batch_queries, [])
        if isinstance(batch_passages[0], list):
            batch_passages = sum(batch_passages, [])

        batch_student_query = self.tokenizer.pad(
            batch_student_query,
            padding='max_length',
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        batch_student_passage = self.tokenizer.pad(
            batch_student_passage,
            padding='max_length',
            max_length=self.max_passage_length,
            return_tensors="pt",
        )

        # LM teacher inputs
        batch_queries = self.teacher_tokenizer.batch_encode_plus(
            batch_queries,
            max_length=self.max_teacher_query_length,
            padding=True,
            return_tensors='pt',
            truncation="only_first",
        )
        batch_query_ids = batch_queries["input_ids"]
        batch_query_mask = batch_queries["attention_mask"]

        batch_passages = self.teacher_tokenizer.batch_encode_plus(
            batch_passages,
            max_length=self.max_teacher_passage_length,
            truncation='only_first',
            padding=True,
            return_tensors='pt'
        )
        batch_passages['labels'] = batch_query_ids

        return batch_student_query, batch_student_passage, batch_passages, batch_query_mask
