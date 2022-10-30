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
    rand_idx = random.sample(range(len(dataset)), 10)
    sampled_features = [dataset[idx] for idx in rand_idx]
    collated_features = data_collator(sampled_features)
    logger.info(collated_features.keys())
    batch_size, num_sent, seq_len = collated_features['input_ids'].shape
    for b_idx in range(batch_size):
        logger.info('attention mask')
        logger.info(collated_features['attention_mask'][b_idx])
        logger.info('input ids')
        logger.info(collated_features['input_ids'][b_idx])
        for s_idx in range(num_sent):
            logger.info(tokenizer.decode(collated_features['input_ids'][b_idx][s_idx], skip_special_tokens=False))
        if 'labels' in collated_features:
            logger.info('labels')
            labels = [(l if l >= 0 else tokenizer.pad_token_id) for l in collated_features['labels'][b_idx]]
            logger.info(labels)
            logger.info(tokenizer.decode(labels, skip_special_tokens=False))
        for s_idx in range(num_sent):
            if 'decoder_input_ids' in collated_features:
                logger.info('decoder_input_ids')
                # decoder_input_ids = [(l if l >= 0 else tokenizer.pad_token_id) for l in collated_features['decoder_input_ids'][b_idx][s_idx]]
                logger.info(collated_features['decoder_input_ids'][b_idx][s_idx])
                logger.info(tokenizer.decode(collated_features['decoder_input_ids'][b_idx][s_idx], skip_special_tokens=False))
        logger.info('==============================================')
