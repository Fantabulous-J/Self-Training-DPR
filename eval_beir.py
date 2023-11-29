import argparse
import logging
from typing import List, Dict, Union

import numpy as np
import torch.multiprocessing as mp
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from datasets import Dataset
from sentence_transformers import SentenceTransformer, models
from torch import Tensor

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


class SentenceBERT:
    def __init__(self, model, sep: str = " "):
        self.sep = sep

        self.q_model = model
        self.doc_model = model

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker,
                            args=(process_id, device_name, self.doc_model, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool['output']
        [output_queue.get() for _ in range(len(pool['processes']))]
        return self.doc_model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[
        List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> \
            Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)

    # Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str],
                               batch_size: int = 8, chunk_id: int = None, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]

        if chunk_id is not None and chunk_id >= len(pool['processes']):
            output_queue = pool['output']
            output_queue.get()

        input_queue = pool['input']
        input_queue.put([chunk_id, batch_size, sentences])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=350)
    parser.add_argument('--pooling_mode', type=str, default='cls')
    args = parser.parse_args()

    data_dir = args.data_dir
    task = args.task
    checkpoint_path = args.checkpoint_path
    max_seq_length = args.max_seq_length
    pooling_mode = args.pooling_mode

    assert task in ['fiqa', 'scifact', 'arguana', 'climate-fever', 'dbpedia-entity', 'cqadupstack', 'quora', 'scidocs',
                    'fever', 'nfcorpus', 'trec-covid', 'webis-touche2020', 'hotpotqa', 'nq', 'robust04', 'trec-news',
                    'signal1m', 'bioasq'], "Invalid task {}".format(task)

    logging.info("###### {} ######".format(task))

    if task in ['fever', 'webis-touche2020']:
        max_seq_length = 128
    logger.info("Load model from {}, max_seq_length {}".format(checkpoint_path, max_seq_length))
    bert = models.Transformer(checkpoint_path, max_seq_length=max_seq_length)
    pool = models.Pooling(bert.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    model = SentenceTransformer(modules=[bert, pool])
    model = DRES(SentenceBERT(model, sep='[SEP]'), batch_size=16)

    if task == 'cqadupstack':
        ndcgs = []
        _maps = []
        recalls = []
        precisions = []

        ce_ndcgs = []
        ce_maps = []
        ce_recalls = []
        ce_precisions = []

        for subtask in ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex',
                        'unix', 'webmasters', 'wordpress']:
            corpus, queries, qrels = GenericDataLoader(data_folder='{}/{}/{}'.format(data_dir, task, subtask)).load(
                split="test")

            retriever = EvaluateRetrieval(model, score_function="dot")
            results = retriever.retrieve(corpus, queries)

            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            ndcgs.append(ndcg)
            _maps.append(_map)
            recalls.append(recall)
            precisions.append(precision)

        ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
        _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
        recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
        precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
        for eval in [ndcg, _map, recall, precision]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder='{}/{}'.format(data_dir, task)).load(split="test")

        retriever = EvaluateRetrieval(model, score_function="dot")
        results = retriever.retrieve(corpus, queries)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    logging.info('\n')

