from typing import Optional, Union, Any, List
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    model_type: str = 'encoder'
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        special_keys = ['input_ids', 'attention_mask']

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        assert labels is None

        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        features = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features = {k: features[k].view(bs, num_sent, -1) if k in special_keys else features[k].view(bs, num_sent, -1)[:, 0] for
                    k in features}
        if self.model_type == 'seq2seq':
            features["decoder_input_ids"] = torch.full([bs, num_sent, 1], self.tokenizer.pad_token_id, dtype=torch.long)

        # prepare decoder_input_ids
        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and 'labels' in features and self.model_type == 'seq2seq':
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        return features
