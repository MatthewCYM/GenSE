import logging
import datasets
import transformers
import sys
import random


logger = logging.getLogger(__name__)


def setup_logging(log_level):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def show_examples(dataset, data_collator, tokenizer):
    rand_idx = random.sample(range(len(dataset)), 2)
    sampled_features = [dataset[idx] for idx in rand_idx]
    collated_features = data_collator(sampled_features)
    logger.info(collated_features.keys())
    for idx in range(collated_features['input_ids'].shape[0]):
        logger.info(collated_features['input_ids'][idx])
        if 'labels' in collated_features:
            logger.info(collated_features['labels'][idx])
        logger.info(tokenizer.decode(collated_features['input_ids'][idx], skip_special_tokens=False))
        if 'labels' in collated_features:
            # handle -100
            labels = [(l if l >= 0 else tokenizer.pad_token_id) for l in collated_features['labels'][idx]]
            logger.info(tokenizer.decode(labels, skip_special_tokens=False))
        logger.info('==============================================')
