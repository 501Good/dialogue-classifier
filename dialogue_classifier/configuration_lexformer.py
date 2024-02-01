from typing import Optional
from transformers import PretrainedConfig, AutoConfig


class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    Examples:
    ```python
    >>> from transformers import BertConfig, BertModel
    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = BertConfig()
    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


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
            raise ValueError(f"`loss_reduction` must be 'mean', 'sum' or 'none', got {loss_reduction}.")

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