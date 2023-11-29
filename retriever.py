from typing import Union

import numpy as np
import faiss
import torch
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import gc

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# def my_index_cpu_to_gpu_multiple(resources, index, co=None, gpu_nos=None):
#     vres = faiss.GpuResourcesVector()
#     vdev = faiss.IntVector()
#     if gpu_nos is None:
#         gpu_nos = range(len(resources))
#     for i, res in zip(gpu_nos, resources):
#         vdev.push_back(i)
#         vres.push_back(res)
#     index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
#     index.referenced_objects = resources
#     return index


class FaissIPRetriever:
    def __init__(self, init_reps: np.ndarray, use_gpu=False):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        if use_gpu:
            ngpus = faiss.get_num_gpus()
            assert ngpus > 0, ngpus
            config = faiss.GpuMultipleClonerOptions()
            config.shard = True
            config.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=config)
            # if ngpus == 2:
            #     index = faiss.index_cpu_to_all_gpus(index, co=config)
            # else:
            #     gpu_nos = [2, 3]
            #     resources = [faiss.StandardGpuResources() for i in gpu_nos]
            #     index = my_index_cpu_to_gpu_multiple(resources, index, co=config, gpu_nos=gpu_nos)
        self.index = index
        self.use_gpu = use_gpu

    def search(self, query_vector: np.ndarray, k: int):
        return self.index.search(query_vector, k)

    def add(self, passage_vector: np.ndarray):
        self.index.add(passage_vector)

    def batch_search(self, query_vector: np.ndarray, k: int, batch_size: int):
        num_query = query_vector.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size)):
            nn_scores, nn_indices = self.search(query_vector[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    # psg_indices = [[int(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    # psg_indices = np.array(psg_indices)
    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    return all_scores, psg_indices


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_embeddings', required=True)
    parser.add_argument('--passage_embeddings', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()

    faiss.omp_set_num_threads(72)

    if '*' in args.passage_embeddings:
        index_files = glob.glob(args.passage_embeddings)
        logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

        p_reps_0, p_lookup_0 = torch.load(index_files[0])
        retriever = FaissIPRetriever(p_reps_0.float().numpy(), args.use_gpu)

        shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    else:
        p_reps, p_lookup = torch.load(args.passage_embeddings)
        shards = [(p_reps, p_lookup)]
        retriever = FaissIPRetriever(p_reps.float().numpy(), args.use_gpu)
    look_up = []

    if args.use_gpu:
        all_p_reps = []
        for p_reps, p_lookup in shards:
            all_p_reps.append(p_reps.numpy())
            look_up += p_lookup
        retriever.add(np.concatenate(all_p_reps, axis=0))
    else:
        for p_reps, p_lookup in shards:
            retriever.add(p_reps.float().numpy())
            look_up += p_lookup

    if '*' in args.query_embeddings:
        index_files = glob.glob(args.query_embeddings)
        q_reps_0, q_lookup_0 = torch.load(index_files[0])
        shards = chain([(q_reps_0, q_lookup_0)], map(torch.load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
        all_q_reps, all_q_lookup = [], []
        for q_reps, q_lookup in shards:
            all_q_reps.append(q_reps.float().numpy())
            all_q_lookup += q_lookup
        q_reps = np.concatenate(all_q_reps, axis=0)
        q_lookup = all_q_lookup
    else:
        q_reps, q_lookup = torch.load(args.query_embeddings)
        q_reps = q_reps.float().numpy()

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)
    logger.info('Index Search Finished')

    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        torch.save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()