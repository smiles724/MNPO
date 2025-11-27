import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from scripts.simpo_trainer import SimPOTrainer
from mnpo_scripts.mnpo_config import MNPOConfig
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer


class MNPOTrainer(SimPOTrainer):
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[MNPOConfig] = None,
            **kwargs,
    ):

        super().__init__(model=model, args=args, **kwargs)

        # MNPO parameters
        self.ratio = float(args.ratio)
        self.eta = float(args.eta)
        self.beta = float(args.beta)
        self.max_history_t = int(args.max_history_t)
        if getattr(args, "history_weights", None):
            self.weights = list(args.history_weights)
        else:
            self.weights = [1.0 for _ in range(self.max_history_t)]

    def mnpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            history_logps_list: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        device = self.accelerator.device

        policy_chosen_logps = policy_chosen_logps.to(device=device, dtype=torch.float32)
        policy_rejected_logps = policy_rejected_logps.to(device=device, dtype=torch.float32)
        reference_chosen_logps = reference_chosen_logps.to(device=device, dtype=torch.float32)
        reference_rejected_logps = reference_rejected_logps.to(device=device, dtype=torch.float32)

        pi_logratios = policy_chosen_logps - policy_rejected_logps  # (B,)
        ref_logratios = reference_chosen_logps - reference_rejected_logps  # (B,)

        t = self.max_history_t

        weighted_logratios = torch.zeros_like(pi_logratios, device=device)

        weights = [float(w) for w in self.weights] if hasattr(self, "weights") else []

        if history_logps_list and t > 0:
            effective_t = min(len(history_logps_list), t)

            if effective_t > 0:
                if not weights:
                    weights = [1.0] * effective_t

                weights = weights[:effective_t]
                total_weight = float(sum(weights)) if sum(weights) != 0 else 0.0

                for j, (chosen_j, rejected_j) in enumerate(history_logps_list[:effective_t]):
                    chosen_j = torch.as_tensor(chosen_j, device=device, dtype=torch.float32)
                    rejected_j = torch.as_tensor(rejected_j, device=device, dtype=torch.float32)

                    if total_weight > 0:
                        lambda_j = weights[j] / total_weight
                    else:
                        lambda_j = 1.0 / float(effective_t)

                    weighted_logratios = weighted_logratios + lambda_j * (chosen_j - rejected_j)

        ratio = float(self.ratio)
        eta = float(self.eta)
        beta = float(self.beta)

        logits = pi_logratios - ratio * ref_logratios - (1.0 - ratio) * weighted_logratios

        logits_shift = 1.0 / (2.0 * eta)
        losses = (logits - logits_shift) ** 2

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def pack_history_logps_from_dataset(self, batch: Dict[str, torch.Tensor]) -> List[
        Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Helper function to extract historical logps from the batch.
        """
        history_logps_list = []
        for j in range(self.max_history_t):
            key_c = f"history{j}_chosen_logps"
            key_r = f"history{j}_rejected_logps"
            if key_c in batch and key_r in batch:
                history_logps_list.append((batch[key_c], batch[key_r]))
            else:
                # Stop if we can't find the next history entry
                break
        return history_logps_list

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: str = "train",
    ):
        """
        Compute the MNPO loss and other metrics for the given batch.
        This method is overridden from SimPOTrainer.
        """
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # 1. Get policy logps using the efficient concatenated_forward from SimPOTrainer
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,  # SimPOTrainer's forward returns this, useful for SFT loss
        ) = self.concatenated_forward(model, batch)

        # 2. Get reference and history logps from the pre-computed batch data
        # Ensure they are on the correct device
        reference_chosen_logps = batch['reference_chosen_logps'].to(self.accelerator.device)
        reference_rejected_logps = batch['reference_rejected_logps'].to(self.accelerator.device)

        history_logps_list = self.pack_history_logps_from_dataset(batch)
        history_logps_list = [
            (c.to(self.accelerator.device), r.to(self.accelerator.device))
            for c, r in history_logps_list
        ]

        # 3. Compute MNPO loss
        losses, chosen_rewards, rejected_rewards = self.mnpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            history_logps_list,
        )

        loss = losses.mean()

        # 4. Calculate metrics (same as your original implementation)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return loss, metrics