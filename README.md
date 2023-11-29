# Noisy Self-Training with Synthetic Queries for Dense Retrieval
Source code for our EMNLP 2023 Findings Paper "Noisy Self-Training with Synthetic Queries for Dense Retrieval".

## Install environment
```shell
pip install -r requirements.txt
```

## Evaluation
### Models
- [fanjiang98/STDPR-MSMARCO](https://huggingface.co/fanjiang98/STDPR-MSMARCO): model trained on MS-MARCO.
- [fanjiang98/STDPR-NQ](https://huggingface.co/fanjiang98/STDPR-NQ): model trained on Natural Questions.
- [fanjiang98/STDPR-Beir](https://huggingface.co/fanjiang98/STDPR-Beir): `fanjiang98/STDPR-MSMARCO` adapted on BEIR.
### BEIR
#### Download Dataset
```shell
mkdir -p beir
cd beir
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip
cd ../
```
Other datasets can be downloaded from [here](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/). Note that BioASQ, TREC-NEWS, Robust04 and Signal-1M are not publicly available, please refer to [here](https://github.com/beir-cellar/beir/wiki/Datasets-available) for more details.

#### Retrieval
```shell
python eval_beir.py \
    --data_dir beir \
    --task scifact \
    --checkpoint_path fanjiang98/STDPR-Beir \
    --max_seq_length 350 \
    --pooling_mode cls
```

### MS-MARCO
#### Download Dataset
```shell
wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
cd marco

wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
gunzip qidpidtriples.train.full.2.tsv.gz
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv
```

#### Generate Embeddings
Encode Query
```shell
MODEL_PATH=fanjiang98/STDPR-MSMARCO
DATA_PATH=marco

mkdir -p ${DATA_PATH}/encoding
python encode.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_dir ${DATA_PATH} \
    --fp16 \
    --per_device_eval_batch_size 2048 \
    --encode_is_qry \
    --shared_encoder True \
    --max_query_length 32 \
    --query_file dev.query.txt \
    --dataloader_num_workers 4 \
    --encoded_save_path ${DATA_PATH}/encoding/dev_query_embedding.pt
```
Encode Corpus
```shell
MODEL_PATH=fanjiang98/STDPR-MSMARCO
DATA_PATH=marco

mkdir -p ${DATA_PATH}/encoding
for i in $(seq -f "%02g" 0 9)
do
python encode.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_dir ${DATA_PATH} \
    --fp16 \
    --corpus_file corpus.tsv \
    --shared_encoder True \
    --max_passage_length 128 \
    --per_device_eval_batch_size 512 \
    --encode_shard_index $i \
    --encode_num_shard 10 \
    --dataloader_num_workers 4 \
    --encoded_save_path ${DATA_PATH}/encoding/embedding_split${i}.pt
done
```
#### Retrieve
```shell
DATA_PATH=MARCO

python retriever.py \
    --query_embeddings ${DATA_PATH}/encoding/dev_query_embedding.pt \
    --passage_embeddings ${DATA_PATH}/encoding/'embedding_split*.pt' \
    --depth 1000 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to ${DATA_PATH}/rank.txt
```
We use pyserini for evaluation:
```shell
DATA_PATH=MARCO

python convert_result_to_marco.py \
    --input ${DATA_PATH}/rank.txt \
    --output ${DATA_PATH}/rank.txt.marco

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset checkpoints/${DATA_PATH}/rank.txt.marco

python -m pyserini.eval.trec_eval \
    -c -mrecall.50 \
    -mmap msmarco-passage-dev-subset checkpoints/${DATA_PATH}/rank.txt.marco
```

### Natural Questions
#### Download Dataset
```shell
mkdir -p NQ
cd NQ
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip -d psgs_w100.tsv.gz
cd ../
```

#### Generate Embeddings
Encode Query
```shell
MODEL_PATH=fanjiang98/STDPR-NQ
DATA_PATH=NQ

mkdir -p ${DATA_PATH}/encoding
python encode.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_dir ${DATA_PATH} \
    --fp16 \
    --per_device_eval_batch_size 2048 \
    --encode_is_qry \
    --shared_encoder True \
    --max_query_length 32 \
    --query_file nq-test.qa.csv \
    --dataloader_num_workers 4 \
    --encoded_save_path ${DATA_PATH}/encoding/nq_test_query_embedding.pt
```
Encode Corpus
```shell
MODEL_PATH=fanjiang98/STDPR-NQ
DATA_PATH=NQ

mkdir -p ${DATA_PATH}/encoding
for i in $(seq -f "%02g" 0 19)
do
python encode.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_dir ${DATA_PATH} \
    --fp16 \
    --corpus_file psgs_w100.tsv \
    --shared_encoder True \
    --max_passage_length 128 \
    --per_device_eval_batch_size 512 \
    --encode_shard_index $i \
    --encode_num_shard 20 \
    --dataloader_num_workers 4 \
    --encoded_save_path ${DATA_PATH}/encoding/embedding_split${i}.pt
done
```
#### Retrieve
```shell
DATA_PATH=NQ
python retriever.py \
    --query_embeddings ${DATA_PATH}/encoding/nq_test_query_embedding.pt \
    --passage_embeddings ${DATA_PATH}/encoding/'embedding_split*.pt' \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to ${DATA_PATH}/nq.rank.txt
```
We use pyserini for evaluation:
```shell
python convert_result_to_trec.py --input ${DATA_PATH}/nq.rank.txt --output ${DATA_PATH}/nq.rank.txt.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
    --topics dpr-nq-test \
    --index wikipedia-dpr \
    --input nq.rank.txt.trec \
    --output run.nq.test.json
python -m pyserini.eval.evaluate_dpr_retrieval 
    --retrieval run.nq.test.json  \
    --topk 1 5 20 100
```