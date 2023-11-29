from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    shared_encoder: bool = field(
        default=True,
        metadata={"help": "weight sharing between qry passage encoders"}
    )

    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as teacher_model_name"}
    )
    teacher_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as teacher_model_name"}
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    corpus_file: str = field(default="corpus.tsv", metadata={"help": "corpus text path"})
    query_file: str = field(default="train.query.txt", metadata={"help": "query text path"})
    qrels_file: str = field(default="qrels.train.tsv", metadata={"help": "query passage relation path"})

    labelled_train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    labelled_query_file: str = field(default="train.query.txt", metadata={"help": "query text path"})

    train_n_passages: int = field(default=8)
    no_shuffle_positive: bool = field(default=False)

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=10)
    encode_shard_index: int = field(default=0)
    split: str = field(default='train', metadata={"help": "dataset splits"})

    max_query_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_passage_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_query_passage_length: int = field(
        default=156,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage & query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    retrieval_data_path: str = field(default=None, metadata={"help": "retrieval results data path"})

    max_teacher_query_length: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_teacher_passage_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class BiEncoderTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    qgen_pretrain: bool = field(default=False, metadata={"help": "do pretraining on cleaned synthetic data"})
    retriever_score_scaling: bool = field(default=False, metadata={"help": "scale retriever score or not when "
                                                                           "computing the distribution"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    print_steps: int = field(default=100, metadata={"help": "step for displaying"})
    recover_query: bool = field(default=False, metadata={"help": "recover masked query by using the cls token "
                                                                 "representation for bi-encoder pretraining on "
                                                                 "filtered synthetic queries"})
    num_decoder_layer: int = field(default=2)
    mlm_probability: float = field(default=0.15)

    noisy: bool = field(default=False, metadata={"help": "add noisy self-distillation loss"})
    noise_type: str = field(default='shuffle', metadata={"help": "method to add noises on inputs"})
    noise_prob: float = field(default=0.1)
    multi_task: bool = field(default=False, metadata={"help": "use multi-task learning to train a single model on "
                                                              "BEIR benchmark"})
    target_task: str = field(default=None, metadata={"help": "the target task to be evaluated on for the leave-one-out"
                                                             "multi-task setting"})
    use_cluster: bool = field(default=False, metadata={"help": "the leave-one-out evaluation for multi-task on "
                                                               "datasets of the same cluster"})
    teacher_temp: float = field(default=1)
    student_temp: float = field(default=1)
    kd_weight: float = field(default=0.1, metadata={"help": "weight for self-distillation"})

    intermediate_layers: List[int] = field(default_factory=lambda: [6])
    self_distill: bool = field(default=False)

    self_distill_joint: bool = field(default=False)
    batch_size_ratio: int = field(default=8)

    meta: bool = field(default=False, metadata={"help": "use meta learning to update the teacher based on performance "
                                                        "of the student on a held-out dataset"})
    teacher_learning_rate: float = field(default=3e-6)

    num_experts: int = field(default=9, metadata={"help": "number of experts for multi-task training "
                                                          "on BEIR benchmark"})
    use_instructions: bool = field(default=False, metadata={"help": "add instructions before queries for "
                                                                    "multi-task learning"})
    only_inbatch_negatives: bool = field(default=False, metadata={"help": "only use in-batch negatives"})

    prefix_mlp: bool = field(default=False)
    pre_seq_len: int = field(default=8)
    prefix_hidden_size: int = field(default=512)


@dataclass
class CrossEncoderTrainingArguments(BiEncoderTrainingArguments):
    pass


@dataclass
class PretrainCrossEncoderTrainingArguments(CrossEncoderTrainingArguments):
    print_steps: int = field(default=100, metadata={"help": "step for displaying"})
    qgen_pretrain: bool = field(default=False, metadata={"help": "do pretraining on cleaned synthetic data"})
    mlm_head: bool = field(default=False, metadata={"help": "do MLM during pretraining"})
    condenser: bool = field(default=False, metadata={"help": "use condenser architecture for pretraining"})
    num_condenser_layer: int = field(default=2)
    skip_from: int = field(default=6)
    mlm_probability: float = field(default=0.15)
    enc_dec: bool = field(default=False)
    enc_mlm_probability: float = field(default=0.15)
    dec_mlm_probability: float = field(default=0.15)


@dataclass
class DistilModelArguments(ModelArguments):
    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as teacher_model_name"}
    )
    teacher_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as teacher_model_name"}
    )


@dataclass
class DistilTrainingArguments(BiEncoderTrainingArguments):
    teacher_temp: float = field(default=1)
    student_temp: float = field(default=1)
    learnable_temp: bool = field(default=False)
    distill_embedding: bool = field(default=False)
    rank_consistent: bool = field(default=False)
    print_steps: int = field(default=100, metadata={"help": "step for displaying"})
    noisy_student: bool = field(default=False, metadata={"help": "add noises to student inputs"})
    noise_type: str = field(default='shuffle', metadata={"help": "method to add noises on inputs"})
    noise_prob: float = field(default=0.1)
    noisy_teacher: bool = field(default=False, metadata={"help": "add noises to teacher"})
    multiple_teacher: bool = field(default=False, metadata={"help": "use multiple teachers to generate pseudo labels"})
    feature_align: bool = field(default=False, metadata={"help": "feature alignment (query & passage) between student "
                                                                 "and teacher"})
    margin_mse: bool = field(default=False, metadata={"help": "use margin mse as the loss for knowledge distillation"})
    t5_reranker: bool = field(default=False, metadata={"help": "use t5 encoder as the architecture for reranker"})


@dataclass
class DynamicNegativeDistillTrainingArguments(DistilTrainingArguments):
    teacher_learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for reranker"})
    teacher_weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for reranker if we apply some."})
    update_negative_steps: int = field(default=10000, metadata={"help": "steps for updating negatives using the "
                                                                        "latest retriever"})
    update_negative: bool = field(default=True, metadata={"help": "whether to update hard negatives"})
    passage_embeddings: str = field(default=None, metadata={"help": "path for storing passage embeddings"})
    query_embeddings: str = field(default=None, metadata={"help": "path for storing query embeddings"})
    retrieve_batch_size: int = field(default=5000, metadata={"help": "batch size for faiss search"})
    depth: int = field(default=200, metadata={"help": "top-k negatives"})


@dataclass
class QueryGenTrainingArguments(BiEncoderTrainingArguments):
    pass
