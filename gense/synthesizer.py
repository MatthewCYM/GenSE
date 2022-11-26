from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import logging
import torch


logger = logging.getLogger(__name__)


class Synthesizer(object):
    def __init__(self, model_name_or_path, device: str = None):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device)

    def generate_entailment(
        self,
        input_sents,
        batch_size=128,
    ):
        logger.info('generating entailment')
        prompted_sents = []
        for sent in input_sents:
            prompted_sents.append(
                f'Write two sentences that are entailment. Sentence 1: \"{sent}\"Sentence 2:'
            )
        all_outputs = self.forward(prompted_sents, batch_size)
        return {
            'input_sents': input_sents,
            'entailment_sents': all_outputs
        }

    def generate_contradiction(
        self,
        input_sents,
        batch_size=128,
    ):
        logger.info('generating contradiction')
        prompted_sents = []
        for sent in input_sents:
            prompted_sents.append(
                f'Write two sentences that are contradictory. Sentence 1: \"{sent}\"Sentence 2:'
            )
        all_outputs = self.forward(prompted_sents, batch_size)
        return {
            'input_sents': input_sents,
            'contradiction_sents': all_outputs
        }

    def generate_triplets(
        self,
        input_sents,
        batch_size=128,
    ):
        logger.info('generating triplets')
        entail_sents = self.generate_entailment(input_sents, batch_size)['entailment_sents']
        contra_sents = self.generate_contradiction(input_sents, batch_size)['contradiction_sents']
        return {
            'input_sents': input_sents,
            'entailment_sents': entail_sents,
            'contradiction_sents': contra_sents
        }

    def filter_triplets(
        self,
        input_triplets,
        batch_size=128
    ):
        logger.info('filtering triplets')
        prompted_sents = []
        for sent0, sent1, sent2 in zip(input_triplets['input_sents'], input_triplets['entailment_sents'], input_triplets['contradiction_sents']):
            prompted_sents.append(
                f'if \"{sent0}\", does this mean that \"{sent1}\"? true or false'
            )
            prompted_sents.append(
                f'if \"{sent0}\", does this mean that \"{sent2}\"? true or false'
            )
        all_outputs = self.forward(prompted_sents, batch_size)
        preds0 = [item for idx, item in enumerate(all_outputs) if idx % 2 == 0]
        preds1 = [item for idx, item in enumerate(all_outputs) if idx % 2 == 1]
        rtn = {
            'input_sents': [],
            'entailment_sents': [],
            'contradiction_sents': []
        }
        for sent0, sent1, sent2, pred0, pred1 in \
            zip(input_triplets['input_sents'], input_triplets['entailment_sents'], input_triplets['contradiction_sents'], preds0, preds1):
            if pred0 != 'true' or pred1 != 'false':
                continue
            rtn['input_sents'].append(sent0)
            rtn['entailment_sents'].append(sent1)
            rtn['contradiction_sents'].append(sent2)
        return rtn

    def filter_entailment(
        self,
        input_triplets,
        batch_size=128
    ):
        logger.info('filtering entailment')
        prompted_sents = []
        for sent0, sent1 in zip(input_triplets['input_sents'], input_triplets['entailment_sents']):
            prompted_sents.append(
                f'if \"{sent0}\", does this mean that \"{sent1}\"? true or false'
            )
        all_outputs = self.forward(prompted_sents, batch_size)
        rtn = {
            'input_sents': [],
            'entailment_sents': []
        }
        for sent0, sent1, pred in zip(input_triplets['input_sents'], input_triplets['entailment_sents'],
                                      all_outputs):
            if pred != 'true':
                continue
            rtn['input_sents'].append(sent0)
            rtn['entailment_sents'].append(sent1)
        return rtn

    def filter_contradiction(
        self,
        input_triplets,
        batch_size=128
    ):
        logger.info('filtering contradiction')
        prompted_sents = []
        for sent0, sent1 in zip(input_triplets['input_sents'], input_triplets['contradiction_sents']):
            prompted_sents.append(
                f'if \"{sent0}\", does this mean that \"{sent1}\"? true or false'
            )
        all_outputs = self.forward(prompted_sents, batch_size)
        rtn = {
            'input_sents': [],
            'contradiction_sents': []
        }
        for sent0, sent1, pred in zip(input_triplets['input_sents'], input_triplets['contradiction_sents'], all_outputs):
            if pred != 'false':
                continue
            rtn['input_sents'].append(sent0)
            rtn['contradiction_sents'].append(sent1)
        return rtn

    def forward(
        self,
        prompted_sents,
        batch_size
    ):
        all_outputs = []
        for idx in tqdm(range(len(prompted_sents)//batch_size+1)):
            current_sents = prompted_sents[idx*batch_size: idx*batch_size+batch_size]
            input_features = self.tokenizer(current_sents, add_special_tokens=True, padding=True, return_tensors='pt')
            input_features = {k: v.to(self.device) for k, v in input_features.items()}
            # generation
            outputs = self.model.generate(**input_features, top_p=0.9)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(outputs)
        return all_outputs
