import logging
import sys
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import transformers
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length
import torch.nn as nn
import os

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the precompute script.
    """

    # loss parameters
    beta: Optional[float] = field(default=0.005, metadata={"help": "beta parameter for DPO loss"})

    # model parameters
    model_name_or_path: Optional[str] = field(
        default="sshleifer/tiny-gpt2",
        metadata={"help": "base model name or local path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "reference/SFT model name or local path"},
    )
    last_model: Optional[str] = field(
        default="",
        metadata={"help": "last-iteration model name or path (if any)"},
    )

    # data I/O
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "train dataset path or HF hub id"},
    )
    # Backward-compatible alias: either pass --test_dir or --eval_dir
    test_dir: Optional[str] = field(
        default=None,
        metadata={"help": "test dataset path or HF hub id (optional; if provided, will produce a DatasetDict with 'train' and 'test')"},
    )
    eval_dir: Optional[str] = field(
        default=None,
        metadata={"help": "alias of --test_dir for backward compatibility"},
    )

    # optimization settings (some may be unused in this precompute phase but kept for parity)
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "LoRA rank"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "saving strategy"})
    save_steps: Optional[int] = field(default=50000, metadata={"help": "saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "loss type"})
    output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "train on a small subset (e.g., 100 samples)"})
    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "maximum sample size"})
    choose_type: Optional[str] = field(default="max_min", metadata={"help": "choose type"})

    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'Reporting destinations: "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard", "clearml", "wandb", "all", or "none".'
        },
    )

    # distributed training debug flag
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers; see https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "end-of-text token override"})
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "whether to mask prompt tokens"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "length penalty"})
    history_paths: Optional[List[str]] = field(default_factory=list, metadata={"help": "list of historical model paths"})
    max_history_t: Optional[int] = field(default=2, metadata={"help": "maximum history length"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "cache directory for models and datasets"})


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = True,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Shape (batch, seq_len, vocab)
        labels: Shape (batch, seq_len); tokens == label_pad_token_id are ignored
        average_log_prob: average per non-masked token if True, else sum
        label_pad_token_id: label pad id
        is_encoder_decoder: whether the model is encoder-decoder

    Returns:
        Shape (batch,)
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch, seq_len) and labels must have the same shape on those dims.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # replace pad labels with a dummy id (ignored by loss via mask)
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    padding_value: int = 0,
    label_pad_token_id: int = -100,
) -> Dict[str, torch.LongTensor]:
    """
    Take a batch with separate chosen/rejected tensors and concatenate them.
    """
    concatenated_batch = {}
    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )

    return concatenated_batch


def concatenated_forward(model: nn.Module, batch: Dict) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Core forward pass consistent with DPO Trainer behavior.
    Takes a batch processed by DPODataCollatorWithPadding and returns
    chosen/rejected log-probabilities.
    """
    # 1) concatenate chosen/rejected
    concatenated_batch = concatenated_inputs(batch)

    # 2) prepare model inputs
    input_ids = concatenated_batch["concatenated_input_ids"]
    labels = concatenated_batch["concatenated_labels"]
    attention_mask = concatenated_batch["concatenated_attention_mask"]

    # 3) forward
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # 4) compute logps
    all_logps = get_batch_logps(logits, labels)

    # 5) split back
    bsz = batch["chosen_labels"].shape[0]
    chosen_logps = all_logps[:bsz]
    rejected_logps = all_logps[bsz:]

    return chosen_logps, rejected_logps


def transform_chat_to_str(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert 'chosen' and 'rejected' fields from a list of dicts to a single string.
    Assumes desired content is in the last message of the list.
    """
    if isinstance(example.get("chosen"), list) and example["chosen"]:
        example["chosen"] = example["chosen"][-1]["content"]
    if isinstance(example.get("rejected"), list) and example["rejected"]:
        example["rejected"] = example["rejected"][-1]["content"]
    return example


def tokenize_row(
    feature: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    max_prompt_length: int,
) -> Dict[str, Any]:
    prompt = feature["prompt"]
    chosen_response = feature["chosen"]
    rejected_response = feature["rejected"]

    prompt_tokens = tokenizer(prompt, max_length=max_prompt_length, truncation=True)
    chosen_tokens = tokenizer(prompt + chosen_response, max_length=max_length, truncation=True)
    rejected_tokens = tokenizer(prompt + rejected_response, max_length=max_length, truncation=True)

    chosen_labels = chosen_tokens["input_ids"][:]
    chosen_labels[: len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])
    rejected_labels = rejected_tokens["input_ids"][:]
    rejected_labels[: len(prompt_tokens["input_ids"])] = [-100] * len(prompt_tokens["input_ids"])

    return {
        "prompt": prompt,  # keep original prompt text as well
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
        "rejected_labels": rejected_labels,
    }


def compute_and_add_logps(
    dataset: DatasetDict,
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    accelerator: Accelerator,
    column_prefix: str,
) -> DatasetDict:
    logger.info(f"--- Processing model: {model_path} for columns with prefix: '{column_prefix}' ---")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        cache_dir=args.cache_dir,
    ).eval()
    model = accelerator.prepare_model(model)

    # Use DPODataCollatorWithPadding
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )

    for split in dataset.keys():
        split_dataset = dataset[split]
        dataloader = DataLoader(
            split_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        dataloader = accelerator.prepare(dataloader)

        all_chosen_logps, all_rejected_logps = [], []
        for batch in tqdm(dataloader, desc=f"Computing '{column_prefix}' logps for split={split}"):
            with torch.no_grad():
                chosen_logps, rejected_logps = concatenated_forward(model, batch)

            chosen_logps, rejected_logps = accelerator.gather_for_metrics((chosen_logps, rejected_logps))
            all_chosen_logps.append(chosen_logps.cpu())
            all_rejected_logps.append(rejected_logps.cpu())

        # Add new columns back to the (tokenized) split dataset
        dataset[split] = split_dataset.add_column(
            f"{column_prefix}_chosen_logps", torch.cat(all_chosen_logps).float().numpy()
        )
        dataset[split] = dataset[split].add_column(
            f"{column_prefix}_rejected_logps", torch.cat(all_rejected_logps).float().numpy()
        )

    del model
    accelerator.free_memory()
    torch.cuda.empty_cache()
    return dataset


def load_flexible_dataset(dataset_name_or_path: str, cache_dir: Optional[str] = None, split: str = "train"):
    """
    Load a dataset from a local file/directory or the Hugging Face Hub.

    Args:
        dataset_name_or_path: path to file/dir or a Hub dataset id
        cache_dir: HF cache dir
        split: split name if applicable
    """
    # Case 1: exact file path
    if os.path.isfile(dataset_name_or_path):
        print(f"Detected local file: {dataset_name_or_path}")
        file_type = dataset_name_or_path.split(".")[-1]
        if file_type == "jsonl":
            file_type = "json"  # jsonl uses the 'json' loader
        print(f"Inferred file type '{file_type}'. Loading...")
        try:
            return load_dataset(
                file_type,
                data_files=dataset_name_or_path,
                split=split,
                cache_dir=cache_dir,
            )
        except Exception:
            print("Failed with inferred type; retrying with 'json' loader...")
            return load_dataset(
                "json",
                data_files=dataset_name_or_path,
                split=split,
                cache_dir=cache_dir,
            )

    # Case 2: directory (saved HF dataset) or Hub name
    if os.path.isdir(dataset_name_or_path):
        print(f"Detected local directory: {dataset_name_or_path}; trying to load from disk...")
        loaded_object = load_from_disk(dataset_name_or_path)

        if isinstance(loaded_object, Dataset):
            print("   -> Loaded a single Dataset; returning it directly.")
            return loaded_object
        elif isinstance(loaded_object, DatasetDict):
            print("   -> Loaded a DatasetDict; selecting split...")
            if split in loaded_object:
                return loaded_object[split]
            else:
                available_splits = list(loaded_object.keys())
                raise ValueError(
                    f"Split '{split}' not found in the loaded dataset. Available splits: {available_splits}"
                )
        else:
            raise TypeError(f"Unexpected object type loaded from disk: {type(loaded_object)}")

    # Case 3: Hub id
    print(f"No local path found: {dataset_name_or_path}; loading from the Hugging Face Hub...")
    return load_dataset(dataset_name_or_path, split=split, cache_dir=cache_dir)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, cache_dir=script_args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Load raw train (and optional test) datasets -----
    logger.info(f"Loading initial raw train dataset from: {script_args.train_dir}")
    raw_train = load_flexible_dataset(script_args.train_dir, cache_dir=script_args.cache_dir, split="train")

    # accept --test_dir or legacy --eval_dir
    test_path = script_args.test_dir or script_args.eval_dir
    raw_test = None
    if test_path:
        logger.info(f"Loading raw test dataset from: {test_path}")
        raw_test = load_flexible_dataset(test_path, cache_dir=script_args.cache_dir, split="train")

    # Build a DatasetDict so everything downstream works on both splits
    if raw_test is not None:
        raw_dataset = DatasetDict({"train": raw_train, "test": raw_test})
    else:
        raw_dataset = DatasetDict({"train": raw_train})

    # ----- Normalize chat format to plain strings -----
    logger.info("Transforming 'chosen'/'rejected' columns from list-of-dicts to strings...")
    raw_dataset = raw_dataset.map(transform_chat_to_str, num_proc=12)
    logger.info("Transformation complete.")

    if script_args.sanity_check:
        raw_dataset["train"] = raw_dataset["train"].select(range(min(100, len(raw_dataset["train"]))))
        if "test" in raw_dataset:
            raw_dataset["test"] = raw_dataset["test"].select(range(min(100, len(raw_dataset["test"]))))

    # ----- Tokenize (SimPO-style) -----
    logger.info("Tokenizing dataset in SimPO-style...")
    tokenized_dataset = raw_dataset.map(
        tokenize_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": script_args.max_length,
            "max_prompt_length": script_args.max_prompt_length,
        },
        num_proc=12,
    )
    logger.info("Tokenization complete.")

    # We'll compute logps on the tokenized dataset and store columns on it.
    dataset_with_logps = tokenized_dataset

    # Reference model logps
    if not script_args.ref_model:
        raise ValueError("--ref_model must be provided for precompute.")
    dataset_with_logps = compute_and_add_logps(
        dataset=dataset_with_logps,
        model_path=script_args.ref_model,
        tokenizer=tokenizer,
        args=script_args,
        accelerator=accelerator,
        column_prefix="reference",
    )

    # Historical models logps (optional)
    if script_args.history_paths:
        for i, model_path in enumerate(script_args.history_paths):
            dataset_with_logps = compute_and_add_logps(
                dataset=dataset_with_logps,
                model_path=model_path,
                tokenizer=tokenizer,
                args=script_args,
                accelerator=accelerator,
                column_prefix=f"history{i}",
            )

    # Save final dataset (DatasetDict with train and optionally test)
    if accelerator.is_main_process:
        logger.info(f"Saving final dataset (with logps) to: {script_args.output_dir}")
        dataset_with_logps.save_to_disk(script_args.output_dir)
        logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
