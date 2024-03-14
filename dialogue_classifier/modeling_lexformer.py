# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput

from .configuration_bert import BertConfig
from .configuration_lexformer import LexFormerConfig
from .modeling_bert import (
    PAD_ID,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)


class LexFormerClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float, num_classes: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.relu = nn.LeakyReLU()
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def init_weights(self):
        nn.init.xavier_uniform_(
            self.dense.weight,
            nn.init.calculate_gain("leaky_relu", 0.2),
        )
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(inputs)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


@dataclass
class LexBertOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    lexicon_attentions: Optional[Tuple[torch.FloatTensor]] = None
    combine_ratios: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class LexBertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        lexicon_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_lexicon_attentions: Optional[bool] = False,
        output_combine_ratios: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], LexBertOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        if lexicon_attention_mask is not None:
            extended_lexicon_attention_mask = self.get_extended_attention_mask(
                lexicon_attention_mask, input_shape
            )
        else:
            extended_lexicon_attention_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            lexicon_attention_mask=extended_lexicon_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_lexicon_attentions=output_lexicon_attentions,
            output_combine_ratios=output_combine_ratios,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LexBertOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            lexicon_attentions=encoder_outputs.lexicon_attentions,
            combine_ratios=encoder_outputs.combine_ratios,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LexFormerPhraseLevelEncoder(nn.Module):
    """
    Transformer-based Sentence Level Encoder.

    The architecture is similar to the Tiny-BERT, with the exception of the increased hidden size
    to accommodate for the input embeddings.

    Parameters
    ----------
    pooling : str
        The pooling to use to construct the final output. `"cls"` pooling adds two special tokens to
        simulate BERT's `[CLS]` and `[SEP]` tokens, and returns a `[CLS]` representation as a final output. `"mean"`
        pooling averages all the token embeddings to produce the final output.
    device : torch.device
        Current device of the model.

    Raises
    ------
    NotImplementedError
        If `pooling` argument is not among `["cls", "mean"]`.

    """

    def __init__(self, config: LexFormerConfig, device: torch.device):
        """Initializes the encoder."""
        super().__init__()
        self.hidden_size = config.phrase_hidden_size
        self.encoder_config = BertConfig(
            hidden_size=self.hidden_size,
            intermediate_size=config.phrase_intermediate_size,
            num_hidden_layers=config.phrase_num_hidden_layers,
            num_attention_heads=config.phrase_num_attention_heads,
            add_cross_attention=config.cross_attention,
        )
        self.encoder = LexBertModel(self.encoder_config)
        self.special_tokens = nn.Embedding(2, self.hidden_size)
        self.device = device
        self.pooling = config.pooling
        if self.pooling not in ["cls", "mean"]:
            raise NotImplementedError(
                f"Pooling must be either cls or mean! Got {self.pooling} instead."
            )

    def mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: Union[torch.BoolTensor, torch.LongTensor],
    ) -> torch.Tensor:
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.BoolTensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        lexicon_attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.FloatTensor:
        """
        Performs the forward pass of the model.

        Parameters
        ----------
        inputs_embeds : (batch_size, sequence_length, hidden_size) torch.FloatTensor
            Sequence containing the features of the input sequence.
        attention_mask : (batch_size, sequence_length) torch.BoolTensor
            Mask to avoid performing attention on the padding with `True` in the positions of tokens and `False`
            in the padding positions.
        return_attn : bool, optional
            If set to `True`, attention matrices are returned. Returns `None` otherwise. Default is `False`.

        Returns
        -------
        final_hidden : (batch_size, hidden_size) torch.FloatTensor
            Final hidden state for each input sequence.
        attentions: tuple of (batch_size, num_heads, sequence_length, sequence_length) torch.FloatTensor or None
            Tuple of attention scores for each layer of the encoder. Returns `None` if `return_attn` is `False`.
        """
        if self.pooling == "cls":
            cls_token_ids = torch.zeros(
                inputs_embeds.size(0), dtype=torch.long, device=self.device
            ).unsqueeze(-1)
            sep_token_ids = torch.ones(
                inputs_embeds.size(0), dtype=torch.long, device=self.device
            ).unsqueeze(-1)
            cls_token = self.special_tokens(cls_token_ids)
            sep_token = self.special_tokens(sep_token_ids)
            inputs_embeds = torch.cat((cls_token, inputs_embeds, sep_token), dim=1)
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            lexicon_attention_mask=lexicon_attention_mask,
            output_attentions=return_attn,
            output_lexicon_attentions=return_attn,
            output_combine_ratios=return_attn,
        )
        if self.pooling == "cls":
            final_hidden = outputs[1]
        elif self.pooling == "mean":
            final_hidden = self.mean_pooling(outputs, attention_mask)
        if return_attn:
            attentions = outputs[-3:]
        else:
            attentions = None
        return outputs[0], final_hidden, attentions


class LexFormerModel(PreTrainedModel):
    config_class = LexFormerConfig

    def __init__(self, config: LexFormerConfig):
        super().__init__(config)
        self.word_encoder_model = config.word_encoder_model
        self.pooling = config.pooling

        print(config)

        self.word_encoder_config = AutoConfig.from_pretrained(config.word_encoder_model)
        self.word_encoder = AutoModel.from_config(self.word_encoder_config)
        self.word_output_dim = self.word_encoder.config.hidden_size
        self.word_encoder_batch_size = config.word_encoder_batch_size

        self.phrase_encoder = LexFormerPhraseLevelEncoder(config, self.device)
        self.linear_in_size = self.word_output_dim

        self.clf_binary = LexFormerClassificationHead(
            self.linear_in_size, config.classification_head_dropout, config.num_classes
        )

        if not config.binary_only:
            self.clf_regression = LexFormerClassificationHead(
                self.linear_in_size, config.classification_head_dropout, 1
            )

        self.softmax = nn.LogSoftmax(dim=1)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Union[torch.BoolTensor, torch.LongTensor],
    ) -> torch.FloatTensor:
        # First element of model_output contains all token embeddings
        # token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # TODO: Decide if rewrite the function or import from pytorch_wrapper
    def create_mask_from_length(
        self, length_tensor: torch.Tensor, mask_size: int, zeros_at_end: bool = True
    ) -> torch.BoolTensor:
        """
        Creates a binary mask based on length.

        Parameters
        ----------
        length_tensor : (batch_size) torch.LongTensor
            Tensor containing the lengths.
        mask_size : int
            Specifies the mask size. Usually the largest length.
        zeros_at_end : bool, optional
            Whether to put the zeros of the mask at the end. Default is `True`.

        Returns
        -------
        mask : (batch_size, sequence_length) torch.BoolTensor
            Binary mask.

        """

        if zeros_at_end:
            mask = torch.arange(
                0, mask_size, dtype=torch.int, device=length_tensor.device
            )
        else:
            mask = torch.arange(
                mask_size - 1, -1, step=-1, dtype=torch.int, device=length_tensor.device
            )

        mask = mask.int().view([1] * (len(length_tensor.shape)) + [-1])

        return mask < length_tensor.int().unsqueeze(-1)

    def initialize_encoder_weights_from_pretrained(self):
        self.word_encoder = AutoModel.from_pretrained(self.word_encoder_model)

    def word_encoder_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_lens: Optional[List[int]] = None,
        text_lens: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        attentions = None
        if self.word_encoder_batch_size is not None:
            outputs = []
            attentions = []
            for i in range(
                self.word_encoder_batch_size,
                input_ids.size(0),
                self.word_encoder_batch_size,
            ):
                bstart = i - self.word_encoder_batch_size
                bend = i
                output = self.word_encoder(
                    input_ids=input_ids[bstart:bend],
                    attention_mask=attention_mask[bstart:bend],
                    output_attentions=output_attentions,
                )
                outputs.append(output[0])
            output = self.word_encoder(
                input_ids=input_ids[bend:],
                attention_mask=attention_mask[bend:],
                output_attentions=output_attentions,
            )
            outputs.append(output[0])
            model_output = torch.vstack(outputs)
            if output_attentions:
                attentions.append(output[-1])
            assert model_output.size(0) == input_ids.size(
                0
            ), f"Input sizes do not match! Got output size: {model_output.size()} and {input_ids.size()}."
        else:
            output = self.word_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attentions = output[-1]
            model_output = output[0]
        unpacked_model_output = [
            torch.split(model_output[i], input_len)
            for i, input_len in enumerate(input_lens)
        ]
        unpacked_model_output = [
            torch.sum(x, dim=0) / x.size(0)
            for seq in unpacked_model_output
            for x in seq[1:-1]
        ]
        unpacked_model_output = torch.vstack(unpacked_model_output)
        # print(unpacked_model_output.size())
        # sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        word_outputs = torch.split(unpacked_model_output, text_lens.tolist())
        return word_outputs, model_output, attentions

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_lens: Optional[List[int]] = None,
        text_lens: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        lexicon_attention_mask: Optional[torch.Tensor] = None,
        word_lexicon_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> LexBertOutputWithPoolingAndCrossAttentions:
        word_outputs, model_output, attentions = self.word_encoder_forward(
            input_ids, attention_mask, input_lens, text_lens, output_attentions
        )

        output = pad_sequence(word_outputs, batch_first=True, padding_value=PAD_ID)

        if self.pooling == "cls":
            attention_mask = self.create_mask_from_length(
                text_lens + 2, torch.max(text_lens + 2).item()
            )
        elif self.pooling == "mean":
            attention_mask = self.create_mask_from_length(
                text_lens, torch.max(text_lens).item()
            )
        if word_lexicon_attention_mask is not None:
            mask_lens = torch.split(
                torch.sum(word_lexicon_attention_mask, dim=1), text_lens.tolist()
            )
            mask_lens = torch.sum(pad_sequence(mask_lens, batch_first=True), dim=1)
            encoder_hidden_states = torch.split(
                model_output[0][word_lexicon_attention_mask], mask_lens.tolist()
            )
            encoder_hidden_states = pad_sequence(
                encoder_hidden_states,
                batch_first=True,
                padding_value=PAD_ID,
            )
            encoder_attention_mask = self.create_mask_from_length(
                mask_lens, torch.max(mask_lens).item()
            )
            last_hidden_state, pooled_output, attn_binary = self.phrase_encoder(
                output,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                lexicon_attention_mask=None,
                return_attn=output_attentions,
            )
        else:
            last_hidden_state, pooled_output, attn_binary = self.phrase_encoder(
                output,
                attention_mask,
                lexicon_attention_mask=lexicon_attention_mask,
                return_attn=output_attentions,
            )

        return LexBertOutputWithPoolingAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            attentions=attn_binary,
        )


class MultiLabelLoss(nn.Module):
    def __init__(
        self,
        regularization: bool = True,
        l: float = 0.1,
        intervals: Optional[List[int]] = None,
        reduction: Optional[str] = None,
    ):
        super().__init__()
        if reduction is None:
            reduction = "none" if intervals else "mean"
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        self.l = l
        self.reg = regularization
        self.intervals = intervals

    def forward(self, pred, true_1, true_2):
        loss_1 = self.loss_fn(pred, true_1)
        if self.intervals:
            losses = []
            true_sum = torch.sum(true_1, dim=1)
            if len(self.intervals) == 1:
                losses.append(torch.mean(loss_1[true_sum < self.intervals[0]]))
                losses.append(torch.mean(loss_1[true_sum >= self.intervals[0]]))
            else:
                losses.append(torch.mean(loss_1[true_sum <= self.intervals[0]]))
                for i in range(len(self.intervals) - 1):
                    losses.append(
                        torch.mean(
                            loss_1[
                                (true_sum > self.intervals[i])
                                & (true_sum <= self.intervals[i + 1])
                            ]
                        )
                    )
                losses.append(torch.mean(loss_1[true_sum > self.intervals[-1]]))
            loss_1 = torch.mean(
                torch.hstack([loss for loss in losses if not loss.isnan()])
            )
        loss_2 = torch.dist(torch.sum(pred, dim=1), true_2, p=2)
        if self.reg:
            loss = loss_1 + self.l * loss_2
        else:
            loss = loss_1
        return loss, loss_1, loss_2


class MSLELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        loss = torch.nn.functional.mse_loss(
            torch.log1p(input), torch.log1p(target), reduction=self.reduction
        )
        return loss
        # loss = torch.square(torch.log1p(input) - torch.log1p(target))
        # print(loss)
        # if self.reduction == 'none':
        #     return loss
        # elif self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:
        #     raise ValueError(f"Reduction must be 'mean', 'sum' or 'none', but got '{self.reduction}'!")


class LexFormerForSequenceClassification(PreTrainedModel):
    def __init__(self, config: LexFormerConfig):
        super().__init__(config)
        self.num_labels = config.num_classes
        self.config = config

        self.lexformer = LexFormerModel(config)

        self.word_encoder_config = AutoConfig.from_pretrained(config.word_encoder_model)
        self.linear_in_size = self.word_encoder_config.hidden_size

        self.clf_binary = LexFormerClassificationHead(
            self.linear_in_size, config.classification_head_dropout, config.num_classes
        )

        self.softmax = nn.LogSoftmax(dim=1)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def initialize_encoder_weights_from_pretrained(self):
        self.lexformer.initialize_encoder_weights_from_pretrained()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_lens: Optional[List[int]] = None,
        text_lens: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        lexicon_attention_mask: Optional[torch.Tensor] = None,
        word_lexicon_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.lexformer(
            input_ids,
            attention_mask,
            input_lens,
            text_lens,
            token_type_ids,
            lexicon_attention_mask,
            word_lexicon_attention_mask,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        # print(outputs.pooler_output)
        pred_binary = self.clf_binary(outputs.pooler_output)
        if self.config.multilabel and self.config.regression:
            pred_binary = torch.clamp(pred_binary, min=0.0, max=3.0)

        if self.config.problem_type is None:
            if self.config.multilabel:
                self.config.problem_type == "multi_label_classification"
            elif self.config.regression:
                self.config.problem_type == "regression"
            elif self.config.binary_only:
                self.config.problem_type == "single_label_classification"

        loss = None
        if labels is not None:
            # loss_fct = MultiLabelLoss(reduction=self.config.loss_reduction)
            # loss, _, _ = loss_fct(pred_binary_final, labels, labels.sum(dim=1))
            if self.config.multilabel and self.config.regression:
                loss_fct = nn.SmoothL1Loss(reduction="mean")
                loss = loss_fct(pred_binary, labels)
            elif self.config.multilabel and not self.config.regression:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pred_binary, labels.to(float))
            else:
                raise NotImplementedError(
                    "Loss is not implemented for this type of problem!"
                )
        # print(pred_binary)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pred_binary,
            hidden_states=None,
            attentions=outputs.attentions,
        )


class ZILexFormerForSequenceClassification(PreTrainedModel):
    def __init__(self, config: LexFormerConfig):
        super().__init__(config)
        self.num_labels = config.num_classes
        self.config = config

        self.lexformer = LexFormerModel(config)

        self.word_encoder_config = AutoConfig.from_pretrained(config.word_encoder_model)
        self.linear_in_size = self.word_encoder_config.hidden_size

        self.clf_multilabel = LexFormerClassificationHead(
            self.linear_in_size, config.classification_head_dropout, config.num_classes
        )
        self.clf_binary = LexFormerClassificationHead(
            self.linear_in_size, config.classification_head_dropout, config.num_classes
        )

        if not config.binary_only:
            self.clf_regression = LexFormerClassificationHead(
                self.linear_in_size, config.classification_head_dropout, 1
            )

        self.softmax = nn.LogSoftmax(dim=1)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def initialize_encoder_weights_from_pretrained(self):
        self.lexformer.initialize_encoder_weights_from_pretrained()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_lens: Optional[List[int]] = None,
        text_lens: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        lexicon_attention_mask: Optional[torch.Tensor] = None,
        word_lexicon_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # print(input_ids)
        # print(input_ids.size())
        # print(attention_mask)
        # print(attention_mask.size())
        # print(text_lens)
        # print(input_lens)
        outputs = self.lexformer(
            input_ids,
            attention_mask,
            input_lens,
            text_lens,
            token_type_ids,
            lexicon_attention_mask,
            word_lexicon_attention_mask,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        pred_binary = self.clf_binary(outputs.pooler_output)
        pred_multilabel = self.clf_multilabel(outputs.pooler_output)
        pred_multilabel = torch.clamp(pred_multilabel, min=0.0, max=3.0)

        pred_final = pred_multilabel * torch.round(torch.sigmoid(pred_binary))

        if self.config.problem_type is None:
            if self.config.multilabel:
                self.config.problem_type == "multi_label_classification"
            if self.config.regression:
                self.config.problem_type == "regression"
            if self.config.binary_only:
                self.config.problem_type == "single_label_classification"

        loss = None
        if labels is not None:
            # loss_fct = MultiLabelLoss(reduction=self.config.loss_reduction)
            # loss, _, _ = loss_fct(pred_binary_final, labels, labels.sum(dim=1))
            labels_bin = (labels > 0).to(torch.float)
            loss_bin_fct = BCEWithLogitsLoss()
            loss_bin = loss_bin_fct(pred_binary, labels_bin)

            loss_multilabel_fct = nn.SmoothL1Loss(reduction="mean")
            loss_multilabel = torch.sqrt(
                loss_multilabel_fct(
                    pred_multilabel[labels > 0], labels[labels > 0].to(torch.float)
                )
            )

            # loss = loss_bin + torch.nan_to_num(loss_multilabel, nan=0.0)

            if loss_multilabel.isnan():
                loss = loss_bin
            else:
                loss = loss_bin + loss_multilabel

        return SequenceClassifierOutput(
            loss=loss,
            logits=pred_final,
            hidden_states=None,
            attentions=outputs.attentions,
        )
