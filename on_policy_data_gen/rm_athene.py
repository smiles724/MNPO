import argparse
import json
import os
from typing import Dict, List, Iterable, Any

import torch
from torch import nn
from tqdm import tqdm

from transformers import (
    LlamaModel,
    LlamaPreTrainedModel,
    AutoTokenizer,
    pipeline,
    TextClassificationPipeline,
)


# ======================
# 1. Athene model
# ======================
class AtheneForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])
        scores = []

        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            if c_inds.numel() == 0:
                c_ind = -1
            else:
                c_ind = c_inds[-1].item()
            scores.append(rewards[i, c_ind])

        scores = torch.stack(scores)
        return {"scores": scores}


# ======================
# 2. Pipeline
# ======================
class AtheneRewardPipeline(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
        """
        inputs:  messages (list[{"role": ..., "content": ...}])
        """
        return_tensors = self.framework  # "pt"

        formatted = self.tokenizer.apply_chat_template(
            inputs,
            tokenize=False,
        )

        formatted = formatted + self.tokenizer.cls_token

        return self.tokenizer(
            formatted,
            return_tensors=return_tensors,
            max_length=4096,
            padding="longest",
            truncation=True,
        )

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        return float(model_outputs["scores"].cpu().float().item())


# ======================
# 3. IO ：json / jsonl
# ======================
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_json(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2 structures：list 或 {"data": [...]} / {"train": [...]}
    if isinstance(data, list):
        for item in data:
            yield item
    elif isinstance(data, dict):

        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    yield item
                break
    else:
        raise ValueError("Unsupported json structure")


def iter_dataset(path: str) -> Iterable[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(path)
    elif ext == ".json":
        return read_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}, only .json / .jsonl are supported.")


# ======================
# 4. scoring
# ======================
def build_messages(prompt: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input json/jsonl file (gemma2_ufb_part1_split1.jsonl)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output jsonl file")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir for model/tokenizer")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for Athene pipeline")
    args = parser.parse_args()

    # ======================
    # 4.1 load model and tokenizer
    # ======================
    print("Loading Athene-RM-8B ...")
    model = AtheneForSequenceClassification.from_pretrained(
        "Nexusflow/Athene-RM-8B",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        device_map="auto",  # supporting multi-gpus
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Nexusflow/Athene-RM-8B",
        cache_dir=args.cache_dir,
    )

    if tokenizer.cls_token is None:
        raise ValueError("Tokenizer has no cls_token, but Athene RM expects one.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    rm_pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        pipeline_class=AtheneRewardPipeline,
        device_map="auto",
    )

    # ======================
    # 4.2 traverse data
    # ======================
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    fout = open(args.output_file, "w", encoding="utf-8")

    data_iter = iter_dataset(args.input_file)

    for sample in tqdm(data_iter, desc="Scoring"):
        # expected key：prompt_id, prompt, all_generated_responses, ...
        prompt_id = sample.get("prompt_id")
        prompt = sample["prompt"]
        responses: List[str] = sample["all_generated_responses"]


        all_messages = [build_messages(prompt, resp) for resp in responses]

        scores: List[float] = rm_pipe(
            all_messages,
            batch_size=args.batch_size,
        )

        import numpy as np

        scores_array = np.array(scores, dtype=float)
        best_idx = int(scores_array.argmax())
        worst_idx = int(scores_array.argmin())

        best_resp = responses[best_idx]
        worst_resp = responses[worst_idx]

        chosen = build_messages(prompt, best_resp)
        rejected = build_messages(prompt, worst_resp)

        out_sample = dict(sample)
        out_sample["all_rm_scores"] = [float(s) for s in scores]
        out_sample["chosen"] = chosen
        out_sample["rejected"] = rejected

        fout.write(json.dumps(out_sample, ensure_ascii=False) + "\n")

    fout.close()
    print("Done! Saved to", args.output_file)


if __name__ == "__main__":
    main()
