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
        self.ratio = args.ratio
        self.eta = args.eta
        self.max_history_t = args.max_history_t
        self.beta = args.beta
        self.weights = args.weights

    def mnpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            history_logps_list: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Computes the MNPO loss.
        This is a direct adaptation of your provided loss function.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        t = self.max_history_t
        weighted_logratios = 0.0
        #  weighting strategy, can be made more flexible if needed

        weights = self.weights

        if history_logps_list and t > 0:
            effective_t = len(history_logps_list)
            if effective_t > 0:
                total_weight = sum(weights[:effective_t])
                for j, (chosen_j, rejected_j) in enumerate(history_logps_list):
                    if j < len(weights):
                        lambda_j = weights[j] / total_weight if total_weight > 0 else 1 / effective_t
                        weighted_logratios += lambda_j * (chosen_j - rejected_j)

        logits = pi_logratios - self.ratio * ref_logratios - (1 - self.ratio) * weighted_logratios

        losses = (logits - 1 / (2 * self.eta)) ** 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

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