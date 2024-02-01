#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on DAIC-WOZ."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from dialogue_classifier.configuration_lexformer import LexFormerConfig
from dialogue_classifier.modeling_lexformer import (
    LexFormerForSequenceClassification,
    LexFormerModel,
    ZILexFormerForSequenceClassification,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset configuration (via the datasets library)."
        },
    )

    problem_type: Optional[str] = field(
        default="multi_target_regression",
        metadata={"help": "Type of the prediction problem."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.problem_type not in [
            "multi_target_regression",
            "multi_label_classification",
        ]:
            raise ValueError(
                f"Problem type must be either 'multi_target_regression' or 'multi_label_classification', got {self.problem_type}"
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    word_encoder_layers_to_freeze: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of layers to freeze in the word-level encoder "
                "during training."
            )
        },
    )
    word_encoder_layers_to_train: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated list of layers to train in the word-level encoder "
                "during training."
            )
        },
    )
    pooling: Optional[str] = field(
        default="mean",
        metadata={
            "help": (
                "Type of pooling to use when aggregating the phrase-level representations. "
                "Must be either 'mean' or 'cls'."
            )
        },
    )
    phrase_hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Output hidden size for the phrase-level encoder."},
    )
    phrase_intermediate_size: Optional[int] = field(
        default=1200,
        metadata={"help": "Intermediate hidden size for the phrase-level encoder."},
    )
    phrase_num_hidden_layers: Optional[int] = field(
        default=4, metadata={"help": "Number of layers in the phrase-level encoder."}
    )
    phrase_num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "Number of attention heads in the phrase-level encoder."},
    )
    classification_head_dropout: Optional[float] = field(
        default=0.5,
        metadata={"help": "Dropout probability for the classification head."},
    )
    num_classes: Optional[int] = field(
        default=8, metadata={"help": "Number of outputs of the classification head."}
    )
    loss_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": "Loss reduction for the Multilabel loss. Must me 'mean', 'cls' or 'none'."
        },
    )
    zero_inflated_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use a model with zero-inflated loss or not."},
    )
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Use Low-Rank Adaptaion (LoRA) during training."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    def __post_init__(self):
        if self.pooling not in ["mean", "cls"]:
            raise ValueError(
                f"Pooling must be either 'mean' or 'cls', got '{self.pooling}'!"
            )
        if self.loss_reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Loss reduction must be either 'mean', 'sum' or 'none', got '{self.loss_reduction}'!"
            )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Fix selecting the best model strategy for MAE score
    if training_args.metric_for_best_model == "mae":
        training_args.greater_is_better = False

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {
        split_: f"data/json/{data_args.dataset_name}/{split_}.jsonl"
        for split_ in ["train", "validation", "test"]
    }
    raw_dataset = load_dataset("json", data_files=data_files)

    def collate_fn(batch):
        input_ids = []
        attention_mask = []
        input_lens = []
        for x in batch:
            final_inputs = []
            final_input = []
            final_attention_masks = []
            final_input_len = []
            final_input_lens = []
            input_len = 0
            max_len = tokenizer.model_max_length

            for line in x["input_ids"]:
                if input_len == 0:
                    final_input.append(tokenizer.cls_token_id)
                    input_len += 1
                    final_input_len.append(1)
                if input_len + len(line) + 1 <= max_len - 1:
                    final_input.extend(line[: max_len - 2] + [tokenizer.sep_token_id])
                    final_input_len.append(len(line[: max_len - 2]) + 1)
                    input_len += len(line[: max_len - 2]) + 1
                else:
                    final_inputs.append(
                        final_input
                        + [tokenizer.pad_token_id] * (max_len - len(final_input))
                    )
                    final_input_lens.append(
                        final_input_len + [max_len - len(final_input)]
                    )
                    final_attention_masks.append(
                        [1] * input_len + [0] * (max_len - input_len)
                    )
                    final_input = (
                        [tokenizer.cls_token_id]
                        + line[: max_len - 2]
                        + [tokenizer.sep_token_id]
                    )
                    final_input_len = [1, len(line[: max_len - 2]) + 1]
                    input_len = len(line[: max_len - 2]) + 2
            final_inputs.append(
                final_input + [tokenizer.pad_token_id] * (max_len - len(final_input))
            )
            final_attention_masks.append([1] * input_len + [0] * (max_len - input_len))
            final_input_lens.append(final_input_len + [max_len - len(final_input)])
            input_ids.extend(final_inputs)
            attention_mask.extend(final_attention_masks)
            input_lens.extend(final_input_lens)

        labels = torch.tensor([x["labels"] for x in batch])
        try:
            input_ids = torch.tensor(input_ids)
        except ValueError as e:
            print([len(x) for x in input_ids])
            print(input_ids)
            raise e
        attention_mask = torch.tensor(attention_mask)
        text_lens = torch.tensor([len(x["input_ids"]) for x in batch])
        return {
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_lens": input_lens,
            "text_lens": text_lens,
        }

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if data_args.problem_type == "multi_target_regression":
        config_multilabel = True
        config_regression = True
    elif data_args.problem_type == "multi_label_classification":
        config_multilabel = True
        config_regression = False
    else:
        raise NotImplementedError(f"Unknown problem type {data_args.problem_type}")

    config = LexFormerConfig(
        word_encoder_model=model_args.model_name_or_path,
        num_classes=model_args.num_classes,
        pooling=model_args.pooling,
        phrase_hidden_size=model_args.phrase_hidden_size,
        phrase_intermediate_size=model_args.phrase_intermediate_size,
        phrase_num_attention_heads=model_args.phrase_num_attention_heads,
        phrase_num_hidden_layers=model_args.phrase_num_hidden_layers,
        classification_head_dropout=model_args.classification_head_dropout,
        loss_reduction=model_args.loss_reduction,
        multilabel=config_multilabel,
        regression=config_regression,
        use_lora=model_args.use_lora,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.zero_inflated_model:
        model = ZILexFormerForSequenceClassification(config)
    else:
        model = LexFormerForSequenceClassification(config)
    model = model.to(training_args.device)

    # Preprocessing the raw_datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_dataset = raw_dataset.map(
            lambda examples: tokenizer(
                examples["turns"],
                padding="do_not_pad",
                truncation=True,
                add_special_tokens=False,
            ),
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        raw_dataset.set_format(
            type=None, columns=["turns", "input_ids", "attention_mask", "labels"]
        )
    if training_args.do_train:
        if "train" not in raw_dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_dataset["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]['turns']}."
            )

    # Get the metric function
    metric_f1 = evaluate.load("f1")
    metric_mae = evaluate.load("mae")
    metric_mse = evaluate.load("mse")
    metric_mcc = evaluate.load("matthews_correlation", config_name="multilabel")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    if data_args.problem_type == "multi_target_regression":

        def compute_metrics(p: EvalPrediction):
            predictions, labels = p
            predictions_sum = np.sum(predictions, axis=1)
            labels_sum = np.sum(labels, axis=1)
            predictions = np.array(np.sum(predictions, axis=1) > 9, dtype=np.int32)
            labels = np.array(np.sum(labels, axis=1) > 9, dtype=np.int32)
            return {
                "f1_macro": metric_f1.compute(
                    predictions=predictions, references=labels, average="macro"
                )["f1"],
                "f1_micro": metric_f1.compute(
                    predictions=predictions, references=labels, average="micro"
                )["f1"],
                "mae": metric_mae.compute(
                    predictions=predictions_sum, references=labels_sum
                )["mae"],
                "mse": metric_mse.compute(
                    predictions=predictions_sum, references=labels_sum
                )["mse"],
            }

    elif data_args.problem_type == "multi_label_classification":

        def compute_metrics(p: EvalPrediction):
            predictions, labels = p
            predictions = torch.sigmoid(torch.tensor(predictions))
            labels = torch.tensor(labels)
            metrics = {}
            for threshold in [0.5, 0.7, 0.9]:
                metrics[f"mcc_{threshold}"] = metric_mcc.compute(
                    predictions=(predictions > threshold).to(int), references=labels
                )["matthews_correlation"]
            return metrics

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    data_collator = collate_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            logger.info(
                f"Initializing word encoder weights from {model_args.model_name_or_path} on device {training_args.device}..."
            )
            model.initialize_encoder_weights_from_pretrained()
            model = model.to(training_args.device)

        if model_args.word_encoder_layers_to_freeze is not None:
            logger.info(
                f"Freezing the following layers in the word-level encoder: {model_args.word_encoder_layers_to_freeze}"
            )
            layers_to_freeze = model_args.word_encoder_layers_to_freeze.split(",")
            for name, param in model.lexformer.word_encoder.named_parameters():
                if any([True for layer in layers_to_freeze if layer in name]):
                    param.requires_grad_(False)
            trainable_params = 0
            all_param = 0
            for name, param in model.lexformer.word_encoder.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
        if (
            model_args.word_encoder_layers_to_train is not None
            and not model_args.use_lora
        ):
            logger.info(
                f"Keeping the following layers unfreezed in the word-level encoder: {model_args.word_encoder_layers_to_train}"
            )
            layers_to_train = model_args.word_encoder_layers_to_train
            for name, param in model.lexformer.word_encoder.named_parameters():
                if not re.fullmatch(layers_to_train, name):
                    param.requires_grad_(False)
            trainable_params = 0
            all_param = 0
            for name, param in model.lexformer.word_encoder.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    print(name)
                    trainable_params += param.numel()
            # print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        if model_args.use_lora:
            peft_config = LoraConfig(
                target_modules=model_args.word_encoder_layers_to_train,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model.lexformer.word_encoder = get_peft_model(
                model.lexformer.word_encoder, peft_config
            )
            model = model.to(training_args.device)
            model.lexformer.word_encoder.print_trainable_parameters()
            for name, param in model.lexformer.word_encoder.named_parameters():
                if param.requires_grad:
                    print(name)

        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
            ignore_keys_for_eval=["turns", "id", "speaker_ids"],
        )
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        ).predictions

        output_predict_file = os.path.join(
            training_args.output_dir, f"predict_results_{task}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    print(index, item)
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    kwargs["language"] = "en"
    kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
