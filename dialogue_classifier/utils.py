import torch


class UtteranceCollator:
    def __init__(self, tokenizer, device=torch.device("cpu")):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
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
            max_len = self.tokenizer.model_max_length

            for line in x["input_ids"]:
                if input_len == 0:
                    final_input.append(self.tokenizer.cls_token_id)
                    input_len += 1
                    final_input_len.append(1)
                if input_len + len(line) + 1 <= max_len - 1:
                    final_input.extend(
                        line[: max_len - 2] + [self.tokenizer.sep_token_id]
                    )
                    final_input_len.append(len(line[: max_len - 2]) + 1)
                    input_len += len(line[: max_len - 2]) + 1
                else:
                    final_inputs.append(
                        final_input
                        + [self.tokenizer.pad_token_id] * (max_len - len(final_input))
                    )
                    final_input_lens.append(
                        final_input_len + [max_len - len(final_input)]
                    )
                    final_attention_masks.append(
                        [1] * input_len + [0] * (max_len - input_len)
                    )
                    final_input = (
                        [self.tokenizer.cls_token_id]
                        + line[: max_len - 2]
                        + [self.tokenizer.sep_token_id]
                    )
                    final_input_len = [1, len(line[: max_len - 2]) + 1]
                    input_len = len(line[: max_len - 2]) + 2
            final_inputs.append(
                final_input
                + [self.tokenizer.pad_token_id] * (max_len - len(final_input))
            )
            final_attention_masks.append([1] * input_len + [0] * (max_len - input_len))
            final_input_lens.append(final_input_len + [max_len - len(final_input)])
            input_ids.extend(final_inputs)
            attention_mask.extend(final_attention_masks)
            input_lens.extend(final_input_lens)

        labels = torch.tensor([x["labels"] for x in batch], device=self.device)
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        text_lens = torch.tensor(
            [len(x["input_ids"]) for x in batch], device=self.device
        )
        return {
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_lens": input_lens,
            "text_lens": text_lens,
        }
