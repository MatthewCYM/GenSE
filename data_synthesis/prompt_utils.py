from transformers import PreTrainedTokenizerBase


class PromptTemplate:
    def __init__(
        self,
        prefix,
        postfix,
        midfix,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.prefix = prefix
        self.postfix = postfix
        self.midfix = midfix
        self.tokenizer = tokenizer

        if self.prefix is not None:
            self.encoded_prefix = tokenizer(
                self.prefix,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )

        if self.postfix is not None:
            self.encoded_postfix = tokenizer(
                self.postfix,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )

        if self.midfix is not None:
            self.encoded_midfix = tokenizer(
                self.midfix,
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )

    def process_encoded(
        self,
        s0,
        s1=None,
    ):
        input_ids = s0['input_ids']
        attention_mask = s0['attention_mask']
        if self.prefix is not None:
            input_ids = self.encoded_prefix['input_ids'] + input_ids
            attention_mask = self.encoded_prefix['attention_mask'] + attention_mask
        if s1 is not None:
            if self.midfix is not None:
                input_ids = input_ids + self.encoded_midfix['input_ids']
                attention_mask = attention_mask + self.encoded_midfix['attention_mask']
            input_ids = input_ids + s1['input_ids']
            attention_mask = attention_mask + s1['attention_mask']
        if self.postfix is not None:
            input_ids = input_ids + self.encoded_postfix['input_ids']
            attention_mask = attention_mask + self.encoded_postfix['attention_mask']
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        attention_mask = attention_mask + [1]
        assert len(input_ids) == len(attention_mask)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

