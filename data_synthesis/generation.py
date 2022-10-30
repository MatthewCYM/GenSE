import os
import sys
import nltk
import logging
from datasets import load_dataset
from utils import setup_logging
from data_collator import DataCollatorForSeq2Seq
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from utils import show_examples
from prompt_utils import PromptTemplate
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from trainer import MyTrainer as Seq2SeqTrainer
from arg_utils import ModelArguments, DataTrainingArguments
from arg_utils import MyTrainingArguments as Seq2SeqTrainingArguments
import torch


logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    predict_dataset = load_dataset("text", data_files={'predict': data_args.test_file}, cache_dir=model_args.cache_dir)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )


    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # init prompt templats
    PROMPT_TEMPLATES = {
        'contradiction': PromptTemplate(
            prefix='Write two sentences that are contradictory. Sentence 1: \"',
            postfix='\" Sentence 2:',
            midfix=None,
            tokenizer=tokenizer,
        ),
        'entailment': PromptTemplate(
            prefix='Write two sentences that are entailment. Sentence 1: \"',
            postfix='\" Sentence 2:',
            midfix=None,
            tokenizer=tokenizer,
        ),
        'classification': PromptTemplate(
            prefix='if \"',
            postfix='\"? true or false',
            midfix='\", does this mean that \"',
            tokenizer=tokenizer,
        )
    }

    def preprocess_generation(examples):
        # Get the column names for input/target.
        sent0_column = predict_dataset['predict'].column_names[0]

        sents0 = examples[sent0_column]

        # Avoid "None" fields
        for idx in range(len(sents0)):
            if sents0[idx] is None:
                sents0[idx] = " "


        # we first encoder single sentences
        encoded_sent0 = tokenizer(sents0, padding=False, max_length=data_args.max_source_length, truncation=True,
                                  add_special_tokens=False)

        total_len = len(sents0)
        model_inputs = {
            'input_ids': [],
            'attention_mask': []
        }

        for idx in range(total_len):
            rtn = PROMPT_TEMPLATES['entailment'].process_encoded(
                s0={'input_ids':encoded_sent0['input_ids'][idx], 'attention_mask': encoded_sent0['attention_mask'][idx]}
            )
            model_inputs['input_ids'].append(rtn['input_ids'])
            model_inputs['attention_mask'].append(rtn['attention_mask'])

            rtn = PROMPT_TEMPLATES['contradiction'].process_encoded(
                s0={'input_ids':encoded_sent0['input_ids'][idx], 'attention_mask': encoded_sent0['attention_mask'][idx]}
            )
            model_inputs['input_ids'].append(rtn['input_ids'])
            model_inputs['attention_mask'].append(rtn['attention_mask'])

        return model_inputs


    max_target_length = data_args.val_max_target_length
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset['predict'].map(
            preprocess_generation,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=predict_dataset['predict'].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset_classification=None,
        eval_dataset_generation=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info('The length of predict set is %d' % len(predict_dataset))
    show_examples(predict_dataset, data_collator, tokenizer)

    # Training
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.generate(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                logger.info(len(predictions))
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    return results


if __name__ == "__main__":
    main()
