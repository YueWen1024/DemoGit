# coding=utf-8
# @FileName     : solve_equation.py
# @Time         : 2023/11/30 9:02
# @Author       : YueWen
# @Department   : AILAB
# @Description  : run_clm 简化版本，去掉一些注释和无关的代码
import os
import sys
import math
import torch
import logging
import warnings
import datasets
import evaluate
import transformers
from typing import Optional
from itertools import chain
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry

# 过程：
# 1 大模型：
# 1）训练阶段：构建词表（Sentencepiece）、Train form scratch（CLM）、SFT、RLHF、多机训练（在AutoDL上面实现）
# 2）部署阶段：TensorLLM、CUDA加速、加速原理（Accelerate、DeepSpeed）
# 3）理论阶段：（1）深度学习、机器学习、强化学习
#            （2）Transformers、PyTorch训练框架
#            （3）阅读经典、热门、最新论文
#            （4）LeetCode
# 2 NLP普通模型及常规任务
# 1）BERT系列、T5模型结构
# 2）分类（多标签多分类，单标签多分类）、新词发现（NER）
# 3 机器学习常规任务
# 1）文本聚类
# 白天学习NLP，晚上回去学习英语数学
# 指定日计划、周计划、月计划、年计划


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
                # 出于调试目的或加快训练速度，请将训练样本数截断为此值（如果已设置）。
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # 检测最后的检查点文件
    last_checkpoint = None
    # output_dir 存在且是一个目录 且 do_train为True 且 overwrite_output_dir为False（not False -> True）
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 获取 output_dir 目录下的最后一个检查点文件
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:  # 最后一个检查点文件为空，且 output_dir内容为空
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."  # 使用覆盖输出目录
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                # 检测到检查点，在 last_checkpoint 恢复训练。若要避免此行为，请更改
                # '--output_dir'或添加'--overwrite_output_dir'（覆盖输出目录）从头开始训练。
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:  # 此分支用于从huggingface hub中下载并加载数据集
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:  # 从本地加载数据集文件，数据参数中dataset_name （hub中数据集名称）参数为空
        data_files = {}  # 数据集字典
        dataset_args = {}  # 数据集参数
        if data_args.train_file is not None:  # data_args 中存在 训练集文件（路径）
            data_files["train"] = data_args.train_file  # 训练集文件路径添加到数据集字典中
        if data_args.validation_file is not None:  # data_args 中存在 验证集文件（路径）
            data_files["validation"] = data_args.validation_file  # 验证集文件路径添加到数据集字典中
        # 数据集文件的扩展名
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":  # 若扩展名是txt，则重置扩展名为text
            extension = "text"  # 修改 txt 扩展名为 text
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks  # 是否保留文本中换行符 keep_linebreaks
        raw_datasets = load_dataset(
            extension,  # 数据集文件后缀名，如text、json等
            data_files=data_files,  # 此处传入的是数据集字典
            cache_dir=model_args.cache_dir,  # 数据处理的缓存目录
            token=model_args.token,  # 用于身份验证的令牌（可选），
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        # 若没有验证集，则将使用 validation_split_percentage（验证集划分百分比参数） 来划分数据集。
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(  # 定义并赋值 raw_dataset 字典中的 validation
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",  # 如果 data_args.validation_split_percentage 为 20，则划分为 "train[:20%]"，即前 20% 的数据作为验证集。
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",  # 如果 data_args.validation_split_percentage 为 20，则划分为 "train[20%:]"，即后 80% 的数据作为训练集。
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # 模型配置关键字参数
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:  # 加载hub模型，model_args中存在 config_name 参数
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:  # 加载本地模型（model_args中不存在 config_name 参数 但存在 model_name_or_path 参数）
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:  # config_overrides不为空（用配置覆盖参数）
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # 分词器配置关键字参数
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        # 您正在从头开始实例化新的分词器。此脚本不支持此功能。"
        # 您可以从另一个脚本中完成它，保存它，然后从这里加载它，使用 --tokenizer_name。
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # 创建模型，from_pretrained()、from_config
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # 加载预训练模型权重，
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:  # 不加载预训练模型，使用配置文件，自定义模型，从头开始训练
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # 我们仅在必要时调整嵌入的大小，以避免索引错误。如果要在小词汇上从头开始创建模型，并且想要更小的嵌入大小，请删除此测试。
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.do_train:  # 如果 do_train 为True，则获取训练数据集 (raw_datasets["validation"]) 中的所有列名，并将其转换为一个列表
        column_names = list(raw_datasets["train"].features)  # 即json文件中的key
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]  # 如果"text"在数据集列名列表中

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # 因为这将被保留以避免在tokenize_Function之前加载Hasher强制记录器时出现_LazyModule错误
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # 分词方法
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            # 使用预先定义的分词器 (tokenizer) 对输入的文本进行分词
            # ds['train'][0]: {'text': '毒蕈中毒预防是什么？1.切勿采摘自己不认识的蘑菇食用。2.毫无识别毒蕈经验者，千万不要自采蘑菇。3.预防措施：加强宣传、避免误食。4.有毒野生菇（菌）类常具备以下特征：1）色泽鲜艳度高。2）伞形等菇（菌）表面呈鱼鳞状。3）菇柄上有环状突起物。4）菇柄底部有不规则突起物。5）野生菇（菌）采下或受损，其受损部流出乳汁。'}
            # ds是一个字典
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size.  CLM 输入可能比 block_size 长得多
        if "Token indices sequence length is longer than the" in cl.out:  # “令牌索引序列长度长于”
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
                # 请忽略上面的警告 - 这个长输入将被分成更小的位，然后再传递给模型。
            )
        return output

    # 这是一个上下文管理器，training_args.main_process_first 的作用是确保只有主进程执行其包含的代码块。
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",  # 描述性文本，用于在执行时显示进度条或日志，表示正在对数据集进行分词映射
            )
        else:
            # 分词后的数据集
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:  # 若没有设置block_size
        # Meta llama, tokenizer.model_max_length: 1000000000000000019884624838656
        block_size = tokenizer.model_max_length  # 个人理解：如果model_max_length值是一个极大值，则说明这个模型不限制输入长度
        if block_size > config.max_position_embeddings:  # 若block_size 大于 模型配置config.max_position_embeddings
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )

            block_size = min(1024, config.max_position_embeddings)  # 取1024 和 max_position_embeddings中的最小值
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # 主要的数据处理功能，将连接我们数据集中的所有文本并生成block_size块。
    # 弄清楚这一块的examples的数据类型，样式
    def group_texts(examples):
        # Concatenate all texts.
        # examples[k]列表中的多个字符串元素合并为一个元素，
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  # 比如examples中只有一个key: {"text": []}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            # 将分词后的数据集进行一个块对齐操作（将原始数据分成若干个block_size大小的chunk）
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
        # 经过上述操作会获得一个用于训练语言模型的数据集lm_datasets

    if training_args.do_train:
        if "train" not in tokenized_datasets:  # 是训练但是tokenized_datasets中没有训练集文件，则抛出异常
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:  # 若设置了训练集样本的最大样本数
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)  # 取训练集个数和最大样本数二者中的最小值
            train_dataset = train_dataset.select(range(max_train_samples))  # 在训练集中随机抽取max_train_samples个样本（每个样本大小是block_size）

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        # 对logits进行预处理，以计算指标
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                # 根据模型和配置，logits 可能包含额外的张量，例如 past_key_values，但 logits 始终排在第一位
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):  # 计算指标
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:  # resume_from_checkpoint不为空
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:  # last_checkpoint不为空
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)  # 开始训练
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        # 训练集最大样本数量
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))  # 验证集最大样本数量添加进metrics字典中
        try:
            perplexity = math.exp(metrics["eval_loss"])  # 困惑
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity  # 困惑值添加进metrics字典中

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:  # 如果要将模型推送到hub上
        trainer.push_to_hub(**kwargs)  # 推送到huggingface hub
    else:
        trainer.create_model_card(**kwargs)  # 建一个模型卡片（model card），这是一种描述模型的性能、用途和局限性的文档，通常用于提高模型的透明度和可解释性。


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
