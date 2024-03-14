from typing import Optional

from transformers import AutoConfig, PretrainedConfig


class LexFormerConfig(PretrainedConfig):
    model_type = "lexformer"

    def __init__(
        self,
        word_encoder_model: str = "sentence-transformers/all-distilroberta-v1",
        cross_attention: bool = False,
        pooling: str = "mean",
        phrase_hidden_size: int = 768,
        phrase_intermediate_size: int = 1200,
        phrase_num_hidden_layers: int = 4,
        phrase_num_attention_heads: int = 12,
        classification_head_dropout: float = 0.5,
        word_encoder_batch_size: Optional[int] = None,
        num_classes: int = 8,
        binary_only: bool = False,
        multilabel: bool = True,
        regression: bool = True,
        use_lora: bool = False,
        loss_reduction: str = "mean",
        **kwargs,
    ):
        if pooling not in ["mean", "cls"]:
            raise ValueError(f"`pooling` must be 'mean' or 'cls', got {pooling}.")
        if loss_reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"`loss_reduction` must be 'mean', 'sum' or 'none', got {loss_reduction}."
            )

        self.word_encoder_model = word_encoder_model
        self.cross_attention = cross_attention
        self.pooling = pooling
        self.phrase_hidden_size = phrase_hidden_size
        self.phrase_intermediate_size = phrase_intermediate_size
        self.phrase_num_hidden_layers = phrase_num_hidden_layers
        self.phrase_num_attention_heads = phrase_num_attention_heads
        self.classification_head_dropout = classification_head_dropout
        self.word_encoder_batch_size = word_encoder_batch_size
        self.num_classes = num_classes
        self.binary_only = binary_only
        self.multilabel = multilabel
        self.regression = regression
        self.use_lora = use_lora

        word_encoder_config = AutoConfig.from_pretrained(word_encoder_model)
        self.hidden_size = word_encoder_config.hidden_size

        self.loss_reduction = loss_reduction
        super().__init__(**kwargs)
