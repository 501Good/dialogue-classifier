from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils import ModelOutput


@dataclass
class LexFormerModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.Tensor = None
    past_key_values: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None
    lexicon_attentions: Optional[torch.Tensor] = None
    combine_ratios: Optional[torch.Tensor] = None
