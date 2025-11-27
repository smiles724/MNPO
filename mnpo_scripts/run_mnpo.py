import logging
import random
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel

from mnpo_scripts.mnpo_trainer import MNPOTrainer
from mnpo_scripts.mnpo_config import MNPOConfig
# =====================================================================================
from datasets import load_from_disk

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


# =====================================================================================
# NEW: Utility function for logps calculation (adapted from TRL's DPOTrainer)
# =====================================================================================
def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    labels[labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


# =====================================================================================


def apply_chat_template(
        example,
        tokenizer,
        task: str,
        auto_insert_empty_system_msg: bool = True,
        change_template=None,
):
    # This function remains largely the same, but we will call it for "mnpo" task
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    if task == "mnpo":  # We can reuse the same logic as simpo/dpo
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(f"Require OpenAI format for all messages")

            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(f"Could not format example for `{task}` task!")
    else:
        raise ValueError(f"Task {task} not supported.")
    return example


def main():
    # =====================================================================================
    # MODIFIED: Use MNPOConfig
    # =====================================================================================
    parser = H4ArgumentParser((ModelArguments, DataArguments, MNPOConfig))
    parsed = parser.parse()

    if isinstance(parsed, tuple):
        model_args, data_args, training_args = parsed
    else:
        raise RuntimeError("Expected 3 arguments (Model, Data, Training), got 1.")

    # =====================================================================================
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    if len(data_args.dataset_mixer) > 1:
        raise ValueError("Direct loading only supports a single dataset in dataset_mixer.")

    dataset_path = list(data_args.dataset_mixer.keys())[0]
    logger.info(f"Loading pre-processed dataset directly from disk: {dataset_path}")

    raw_datasets = load_from_disk(dataset_path)

    from datasets import DatasetDict
    if not isinstance(raw_datasets, DatasetDict):
        raw_datasets = DatasetDict({"train": raw_datasets})

    train_dataset = None
    eval_dataset = None
    for split_name in raw_datasets.keys():
        if "train" in split_name:
            train_dataset = raw_datasets[split_name]
        elif "test" in split_name or "eval" in split_name:
            eval_dataset = raw_datasets[split_name]

    if train_dataset is None:
        raise ValueError(
            f"No training split found in the dataset. Available splits: {list(raw_datasets.keys())}"
        )

    logger.info(f"Using '{next(k for k, v in raw_datasets.items() if v is train_dataset)}' for training.")
    if eval_dataset:
        logger.info(f"Using '{next(k for k, v in raw_datasets.items() if v is eval_dataset)}' for evaluation.")
    else:
        logger.info("No evaluation split found or selected.")

    logger.info(f"Loaded dataset splits: {list(raw_datasets.keys())}")
    #####################################
    # Load tokenizer
    #####################################
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    column_names = list(raw_datasets["train"].features)

    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     logger.info(f"Logps sample {index}: {raw_datasets['train'][index]['reference_chosen_logps']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )

    model = model_args.model_name_or_path
    training_args.model_init_kwargs = model_kwargs

    # =====================================================================================
    # Instantiate MNPOTrainer
    # =====================================================================================
    trainer = MNPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    # =====================================================================================

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Add this step to explicitly save the tokenizer.
    if trainer.accelerator.is_main_process:
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook", "mnpo"],  # MODIFIED
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     if "test" in raw_datasets:
    #         metrics["eval_samples"] = len(raw_datasets["test"])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()