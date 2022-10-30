import logging
import os
import sys
import random
import torch
import nltk
from datasets import load_dataset
from utils import setup_logging
from data_collator import DataCollatorForSeq2Seq
from filelock import FileLock
from transformers import (
    T5Config,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from trainer import CustomSeq2SeqTrainer
from arg_utils import ModelArguments, DataTrainingArguments
from arg_utils import MyTrainingArguments as Seq2SeqTrainingArguments
from utils import show_examples
from modeling_t5 import T5ForSentenceEmbedding


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

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
        if extension == 'csv':
            train_dataset = load_dataset(extension, data_files={'train': data_args.train_file}, cache_dir=model_args.cache_dir)
        elif extension == 'tsv':
            train_dataset = load_dataset('csv', data_files={'train': data_args.train_file}, delimiter='\t', cache_dir=model_args.cache_dir)
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
        predict_dataset = load_dataset("text", data_files={'predict': data_args.test_file}, cache_dir=model_args.cache_dir)

    config = T5Config.from_pretrained(
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
    model = T5ForSentenceEmbedding.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_args=model_args,
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

    # we need to check the format of training dataset
    dataset_columns = train_dataset['train'].column_names
    # if len(dataset_columns) == 3:
    #     # triplets
    #     pass
    # else:
    #     raise NotImplementedError

    def preprocess_function_train(examples):
        # Get the column names for input/target.
        dataset_columns = train_dataset['train'].column_names
        sent0_column = dataset_columns[0]
        sent1_column = dataset_columns[1]

        sents0 = examples[sent0_column]
        sents1 = examples[sent1_column]

        if len(dataset_columns) == 3:
            sent2_column = dataset_columns[2]
            sents2 = examples[sent2_column]
        else:
            sents2 = None

        # Avoid "None" fields
        for idx in range(len(sents0)):
            if sents0[idx] is None:
                sents0[idx] = " "
            if sents1[idx] is None:
                sents1[idx] = " "
            if len(dataset_columns) == 3:
                if sents2[idx] is None:
                    sents2[idx] = " "

        total_len = len(sents0)

        inputs = []

        inputs.extend(sents0)
        inputs.extend(sents1)
        if sents2 is not None:
            inputs.extend(sents2)

        # add <cls>
        for i in range(len(inputs)):
            inputs[i] = inputs[i] + ' Question: what can we draw from the above sentence?'

        flatten_features = tokenizer(inputs,
                                     max_length=data_args.max_source_length,
                                     padding=False,
                                     truncation=True,
                                     add_special_tokens=True)



        model_inputs = {}
        for key in flatten_features:
            model_inputs[key] = [
                [flatten_features[key][i+n*total_len] for n in range(2 if sents2 is None else 3)] for i in range(total_len)
            ]

        return model_inputs

    if training_args.do_train:
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset['train'].map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset['train'].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        model_type=model_args.model_type,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info('The length of training set is %d' % len(train_dataset))
    logger.info(train_dataset.column_names)
    show_examples(train_dataset, data_collator, tokenizer)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=False)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
