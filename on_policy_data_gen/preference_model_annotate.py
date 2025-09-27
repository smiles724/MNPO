import json
import argparse
import llm_blender
import numpy as np
import re
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create preference pairs from generated responses using a reward model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the ranker model.")
    return parser.parse_args()


def main(args):
    """Main function to process the data."""
    # 1. Initialize the Blender Ranker Model
    print("Loading llm-blender ranker model...")
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    print("Model loaded successfully.")

    # 2. Load and prepare data for batch processing
    print(f"Reading input file: {args.input_file}")
    prompts_to_rank = []
    candidates_to_rank = []
    original_data = []

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file. {e}")
        return
    except TypeError:
        print(f"Error: Input JSON must be a list of objects.")
        return

    for item in input_list:
        user_prompt = item["prompt"]
        if user_prompt and item.get("all_generated_responses"):
            prompts_to_rank.append(user_prompt)
            candidates_to_rank.append(item["all_generated_responses"])
            original_data.append(item)

    if not prompts_to_rank:
        print("No valid data found to process. Exiting.")
        return

    print(f"Loaded {len(prompts_to_rank)} valid items. Ranking candidates...")

    # 3. Rank all candidates in a single batch call for efficiency
    all_ranks = blender.rank(
        prompts_to_rank,
        candidates_to_rank,
        return_scores=True,
        batch_size=args.batch_size
    )
    print("Ranking complete.")

    # 4. Process rankings and prepare the output
    all_output_records = []
    print("Formatting results...")
    for i in tqdm(range(len(all_ranks)), desc="Formatting Output"):
        scores = all_ranks[i]
        user_prompt = prompts_to_rank[i]
        candidates = candidates_to_rank[i]

        if len(scores) < 2:
            continue

        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)
        chosen_response = candidates[best_idx]
        rejected_response = candidates[worst_idx]

        output_record = {
            "prompt": user_prompt,
            "all_generated_responses": candidates,
            "chosen": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": chosen_response}
            ],
            "rejected": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": rejected_response}
            ]
        }
        # MODIFICATION: Collect all results into a list instead of writing line-by-line
        all_output_records.append(output_record)

    # 5. Write the entire list of results to the output JSON file at once
    print(f"Writing results to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        # Use indent=4 for a readable, "pretty-printed" JSON output
        json.dump(all_output_records, f_out, indent=4, ensure_ascii=False)

    print("Processing finished successfully.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
