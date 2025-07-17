import tyro
from datasets import (
    concatenate_datasets,
    Dataset,
)
from datasets import disable_caching

disable_caching()
# datasets.config.IN_MEMORY_MAX_SIZE = 16_000_000_000

import torch
from transformers.models.speechlm import (
    SpeechLMProcessor,
    SpeechLMForConditionalGeneration,
)
from transformers import TrainingArguments, set_seed
import pdb
from secrets import token_hex
import time
import logging
import os
import torch.distributed as dist
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm
from training_utils import AudioTextDataCollator, CustomTrainer, filter_data
from accelerate.logging import get_logger
from data_utils.utils import get_dataset
import yaml
import psutil
import pandas as pd


# Setup logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = get_logger(__name__)

clean_name = lambda x: x.replace("/", "--")

# set because torch _inductor backend warning said it was a good idea
torch.set_float32_matmul_precision("high")

process = psutil.Process(os.getpid())


def current_cpumem_usage():
    return f"{process.memory_info().rss / 1024 ** 2:.2f}"


def main(config_file: str, dry_run: bool = False):
    ##########################
    ## CONFIGURATION
    ##########################
    # Load the config file
    if config_file is not None:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    targs = TrainingArguments(**config["training_args"])
    targs.output_dir = os.path.expandvars(targs.output_dir)
    audio_encoder = config.get("audio_encoder", None)
    text_decoder = config.get("text_decoder", None)
    ckpt = config.get("ckpt", None)
    datasets = config["datasets"]
    encoder_params = config["encoder_params"]
    decoder_params = config.get("decoder_params", {})
    dataset_workers = int(config["dataset_workers"])
    max_duration = config.get("max_duration", 60)
    min_chars = config.get("min_chars", 3)
    attn_implementation = config.get("attn_implementation", None)
    freeze_encoder = config.get("freeze_encoder", False)
    freeze_decoder = config.get("freeze_decoder", False)
    freeze_adapter = config.get("freeze_adapter", False)
    encoder_lr = float(config.get("encoder_lr", 6e-6))
    decoder_lr = float(config.get("decoder_lr", 2e-5))
    adapter_lr = float(config.get("adapter_lr", 2e-4))
    min_lr_scale = float(config.get("min_lr_scale", 0.1))
    val_samples_per_language = config.get("val_samples_per_language", 1000)
    seed = config.get("seed", 42)
    selected_langs = config.get("selected_langs", None)
    collator_args = config.get("collator_args", {})
    add_pre_adapter = config.get("add_pre_adapter", False)
    num_pre_adapter_layers = config.get("num_pre_adapter_layers", 3)
    ##########################

    # Basic setup
    set_seed(seed)
    is_dist = False
    if dist.is_initialized():
        is_dist = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Distributed setup: rank {rank} out of {world_size}")
    else:
        logger.info("Not in a distributed setup")

    # targs.run_name = (
    # f"{token_hex(2)}_{clean_name(audio_encoder)}_{clean_name(text_decoder)}"
    # )
    logger.info(targs)

    ##########################
    ## DATA LOADING
    ##########################
    train_datasets = list()
    val_datasets = list()
    for dataset_name in datasets:
        logger.info(f"Loading dataset: {dataset_name}")
        data = get_dataset(
            dataset_name,
            splits=["train", "validation"],
            max_duration=max_duration,
            samples_validation=val_samples_per_language,
            selected_langs=selected_langs,
        )

        def add_column(ds, col_name):
            tmp = Dataset.from_dict({col_name: [""] * len(ds)})
            return concatenate_datasets([ds, tmp], axis=1)

        preamble_col = collator_args.get("preamble_col", None)
        if preamble_col is not None and preamble_col not in data["train"].column_names:
            logger.info(f"Adding empty preamble to the dataset: {preamble_col}")
            data["train"] = add_column(data["train"], preamble_col)
            data["validation"] = add_column(data["validation"], preamble_col)
            data["train"] = data["train"].filter(
                lambda x: isinstance(x, str), input_columns=[preamble_col]
            )
            data["validation"] = data["validation"].filter(
                lambda x: isinstance(x, str), input_columns=[preamble_col]
            )

        logger.info(
            f"CPU memory after loading {dataset_name}: {current_cpumem_usage()} MB",
            main_process_only=False,
        )

        if dry_run:
            logger.info("Dry run, using a subset of the dataset")
            data["train"] = data["train"].select(range(1000))

        # filter a subset if it's a dry run
        train_datasets.append(data["train"])
        if targs.do_eval:
            val_datasets.append(data["validation"])

    datasets_count = len(train_datasets)
    logger.info(f"Loaded {datasets_count} training datasets")
    # At this stage, datasets are loaded with columns: audio, text, lang

    ##########################
    ## DATA PREPARATION
    ##########################

    # if ckpt:
    #     processor = SpeechLMProcessor.from_pretrained(ckpt)
    # else:
    processor = SpeechLMProcessor.from_encoder_decoder_pretrained(
        audio_encoder, text_decoder, add_eos_token=True
    )
    # Prepare padding for training
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_dataset = (
        train_datasets[0]
        if datasets_count == 1
        else concatenate_datasets(train_datasets)
    )
    val_dataset = None
    if targs.do_eval:
        val_dataset = (
            val_datasets[0]
            if datasets_count == 1
            else concatenate_datasets(val_datasets)
        )

    ### Preprocess the datasets and set transforms
    logger.info(f"Starting preprocessing... Tracking time")
    stime = time.time()

    # Remove every sample that has a duration longer than the max_duration
    # with targs.main_process_first():
    train_dataset = filter_data(train_dataset, max_duration, min_chars)
    if targs.do_eval:
        val_dataset = filter_data(val_dataset, max_duration, min_chars)

    logger.info(f"Preprocessing took {time.time() - stime:.2f} seconds")
    data_collator = AudioTextDataCollator(processor, **collator_args)
    logger.info(f"Number of rows: {len(train_dataset)}")

    ##########################
    ## MODEL PREPARATION
    ##########################
    # if is_dist and rank == 0:
    #     torch.cuda.memory._record_memory_history(max_entries=100000)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(targs.output_dir)
        and targs.do_train
        and not targs.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(targs.output_dir)
        if last_checkpoint is None and len(os.listdir(targs.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({targs.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and targs.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # TODO: these should be probably optimized with hparam tuning
    if ckpt is not None:
        logger.info(f"Loading model from checkpoint: {ckpt}")

        model = SpeechLMForConditionalGeneration.from_pretrained(
            ckpt, attn_implementation=attn_implementation
        )
    else:
        model_kwargs = {}
        encoder_params = {f"encoder_{k}": v for k, v in encoder_params.items()}
        model_kwargs.update(encoder_params)
        decoder_params = {f"decoder_{k}": v for k, v in decoder_params.items()}
        model_kwargs.update(decoder_params)

        model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs["add_pre_adapter"] = add_pre_adapter
        model_kwargs["num_pre_adapter_layers"] = num_pre_adapter_layers

        model = SpeechLMForConditionalGeneration.from_encoder_decoder_pretrained(
            audio_encoder,
            text_decoder,
            **model_kwargs,
        )
        model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder.pad_token = "<pad>"

        # resize the embedding matrix due to the new task and lang tokens
        model.decoder.resize_token_embeddings(
            len(processor.tokenizer), mean_resizing=False, pad_to_multiple_of=8
        )

    targs.freeze_adapter = freeze_adapter
    targs.freeze_encoder = freeze_encoder
    targs.freeze_decoder = freeze_decoder
    if freeze_adapter:
        logger.info("Freezing the adapter")
        model.freeze_adapter()
    if freeze_encoder:
        logger.info("Freezing the encoder")
        model.freeze_encoder()
    if freeze_decoder:
        logger.info("Freezing the decoder")
        model.freeze_decoder()

    # Print the number of learnable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of learnable parameters: {total_params}")

    ############################
    ## VALUES FOR THE OPTIMIZER
    ############################
    # These values are used within the custom trainer to setup optimizer and scheduler
    targs.encoder_lr = encoder_lr
    targs.decoder_lr = decoder_lr
    targs.adapter_lr = adapter_lr
    targs.min_lr_scale = min_lr_scale

    ##########################
    ## TRAINING
    ##########################

    trainer = CustomTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    if targs.resume_from_checkpoint is not None:
        checkpoint = targs.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    logger.info(f"Train_result: {train_result}")

    ##########################
    ## SAVING COMPONENTS
    ##########################

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if is_dist and rank == 0:
        torch.cuda.memory._dump_snapshot("profile.pkl")
        torch.cuda.memory._record_memory_history(enabled=None)

    trainer.save_model()
    processor.save_pretrained(targs.output_dir)


if __name__ == "__main__":
    tyro.cli(main)
