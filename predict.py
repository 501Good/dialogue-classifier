import json
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from dialogue_classifier.configuration_lexformer import LexFormerConfig
from dialogue_classifier.modeling_lexformer import (
    LexFormerForSequenceClassification,
    ZILexFormerForSequenceClassification,
)


@dataclass
class PredictionArguments:
    model_name: str = field(
        metadata={"help": "The fine-tuned model for making the predictions."}
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset configuration (via the datasets library)."
        }
    )
    data_split: str = field(
        metadata={"help": "Data split to predict on ('validation' or 'test')."}
    )
    output: Optional[str] = field(
        default="predictions.json",
        metadata={"help": "The name of the output JSON file containing predictions."},
    )
    save_attentions: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the output .pt file containing attentions."},
    )
    device: Optional[str] = field(default="cpu", metadata={"help": "Device to use."})


def main():
    parser = HfArgumentParser(PredictionArguments)
    args = parser.parse_args_into_dataclasses()[0]

    device = torch.device(args.device)

    config = LexFormerConfig.from_pretrained(args.model_name)
    if config.architectures[0] == "ZILexFormerForSequenceClassification":
        model = ZILexFormerForSequenceClassification.from_pretrained(
            args.model_name, config=config
        ).to(device)
    else:
        model = LexFormerForSequenceClassification.from_pretrained(
            args.model_name, config=config
        ).to(device)
    model.eval()

    data_files = {
        split_: f"data/json/{args.dataset_name}/{split_}.jsonl"
        for split_ in ["train", "validation", "test"]
    }
    dataset = load_dataset("json", data_files=data_files)

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

        labels = torch.tensor([x["labels"] for x in batch], device=device)
        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = torch.tensor(attention_mask, device=device)
        text_lens = torch.tensor([len(x["input_ids"]) for x in batch], device=device)
        return {
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_lens": input_lens,
            "text_lens": text_lens,
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    encoded_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["turns"],
            padding="do_not_pad",
            truncation=True,
            add_special_tokens=False,
        ),
        batched=False,
        load_from_cache_file=False,
    )

    encoded_dataset.set_format(
        type=None, columns=["input_ids", "attention_mask", "labels"]
    )

    validation_dataloader = torch.utils.data.DataLoader(
        encoded_dataset[args.data_split],
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Get the metric function
    metric_f1 = evaluate.load("f1")
    metric_mae = evaluate.load("mae")
    metric_mse = evaluate.load("mse")
    metric_mcc = evaluate.load("matthews_correlation", config_name="multilabel")

    problem_type = (
        "multi_target_regression"
        if config.regression and config.multilabel
        else "multi_label_classification"
    )

    if problem_type == "multi_target_regression":

        def compute_metrics(p):
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

    elif problem_type == "multi_label_classification":

        def compute_metrics(p):
            predictions, labels = p
            predictions = torch.sigmoid(torch.tensor(predictions))
            labels = torch.tensor(labels)
            metrics = {}
            for threshold in [0.5, 0.7, 0.9]:
                metrics[f"mcc_{threshold}"] = metric_mcc.compute(
                    predictions=(predictions > threshold).to(int), references=labels
                )["matthews_correlation"]
            return metrics

    if args.save_attentions is not None:
        return_attn = True
    else:
        return_attn = False

    preds = []
    labels = []
    attentions = []
    with torch.no_grad():
        for batch in tqdm(
            validation_dataloader,
            desc=f"Predicting on the {args.data_split} set",
            total=len(validation_dataloader),
        ):
            outputs = model(**batch, output_attentions=return_attn)
            preds.append(outputs.logits.tolist())
            labels.append(batch["labels"].tolist())
            if return_attn:
                attentions.append(outputs.attentions)

    if return_attn:
        torch.save(attentions, args.save_attentions)
    json.dump(preds, open(args.output, "w", encoding="utf-8"))

    preds = np.array(preds, dtype=float).squeeze()
    labels = np.array(labels, dtype=float).squeeze()
    p = (preds, labels)
    metrics = compute_metrics(p)
    print(metrics)


if __name__ == "__main__":
    main()
