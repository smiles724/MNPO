from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer, DPOConfig
from accelerate.utils import is_deepspeed_available, tqdm

@dataclass
class PreferenceDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    mask_prompt: Optional[bool] = False

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            # "inputs_ids", "attention_mask" both list
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            # False
            if self.mask_prompt:
                new_attention_mask = [0 for i, p in enumerate(prompt_tokens["attention_mask"])]
            else:
                new_attention_mask = [
                    0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
                ]
            # all 1 if i not in eos_indices_prompt
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt

            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            raise NotImplementedError

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        # 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask'
        # 'prompt', 'chosen', 'rejected', 'chosen_response_only', 'rejected_response_only'
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

class PreComputer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: DPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        mask_prompt: Optional[bool] = False,
        len_penalty: float = 0,
    ):

        if data_collator is None:
            # 2048, 1000, -100, 0, keep_end, None, False
            data_collator = PreferenceDataCollatorWithPadding(
                tokenizer,
                max_length=args.max_length,
                max_prompt_length=args.max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=False,
                max_target_length=max_target_length,
                mask_prompt=mask_prompt,
            )
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        self.use_dpo_data_collator = True
        # self.len_penalty = len_penalty
        self.precompute_ref_log_probs = True
        self._precomputed_train_ref_log_probs = True

    def precompute(self):
        dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

        reference_chosen_logps = []
        reference_rejected_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
            reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
            reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                (reference_chosen_logp, reference_rejected_logp)
            )
            reference_chosen_logps.append(reference_chosen_logp.cpu())
            reference_rejected_logps.append(reference_rejected_logp.cpu())

        all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
        all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

        print(all_reference_chosen_logps.shape, all_reference_rejected_logps.shape)
        return all_reference_chosen_logps, all_reference_rejected_logps