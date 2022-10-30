import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Model
from transformers import AutoConfig, T5Config


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def evaluate_single_checkpoint(
    model_name_or_path,
    pooler,
    mode,
    tasks,
    add_prompt
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    # Load transformers' model checkpoint
    if isinstance(config, T5Config):
        model = T5Model.from_pretrained(model_name_or_path)
    else:
        model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Set params for SentEval
    if mode == 'dev' or mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]
        if add_prompt:
            sentences = [' '.join(s) + ' Question: what can we draw from the above sentence?' for s in batch]
        else:
            sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                # max_length=max_length,
                max_length=512,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True,
            )
        if isinstance(config, T5Config):
            decoder_start_token_id = model._get_decoder_start_token_id()
            batch['decoder_input_ids'] = torch.full([batch['input_ids'].shape[0], 1], decoder_start_token_id)

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)

            last_hidden = outputs.last_hidden_state
            if isinstance(config, T5Config):
                hidden_states = outputs.decoder_hidden_states
            else:
                hidden_states = outputs.hidden_states

        # Apply different poolers
        if pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            pooler_output = outputs.pooler_output
            return pooler_output.cpu()
        elif pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)).cpu()
        elif pooler == "cls_last_two":
            second_last_cls = hidden_states[-2][:, 0]
            last_cls = last_hidden[:, 0]
            avg_cls = (last_cls + second_last_cls) / 2
            return avg_cls.cpu()
        elif pooler == 'cls_last_three':
            third_last_cls = hidden_states[-3][:, 0]
            second_last_cls = hidden_states[-2][:, 0]
            last_cls = last_hidden[:, 0]
            avg_cls = (last_cls + second_last_cls + third_last_cls) / 3
            return avg_cls.cpu()
        elif pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch[
                'attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                            batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if mode == 'dev':
        print("------ %s ------" % (mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif mode == 'test' or mode == 'fasttest':
        print("------ %s ------" % (mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, 
                        choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last', 'cls_last_two', 'cls_last_three'],
                        default='cls',
                        help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
                        default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                                 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                                 'SICKRelatedness', 'STSBenchmark'],
                        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    parser.add_argument('--add_prompt', action='store_true', help="whether to add prompt")

    args = parser.parse_args()

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    evaluate_single_checkpoint(
        model_name_or_path=args.model_name_or_path,
        pooler=args.pooler,
        mode=args.mode,
        tasks=args.tasks,
        add_prompt=args.add_prompt
    )


if __name__ == "__main__":
    main()
