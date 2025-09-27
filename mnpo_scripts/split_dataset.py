
import argparse
import os
from datasets import load_dataset, DatasetDict

def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle & split dataset splits, then save.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Hugging Face dataset repo id (e.g., 'princeton-nlp/gemma2-ultrafeedback-armorm').",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=3,
        help="Number of parts to split each split into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default=os.getenv("MNPO_DATA_PATH", "./data"),
        help="Base directory to save HF datasets (fallback to MNPO_DATA_PATH env or ./data).",
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default=os.getenv("MNPO_JSON_PATH", "./data"),
        help="Directory to save JSONL files (fallback to MNPO_JSON_PATH env or ./data).",
    )
    return parser.parse_args()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def split_data(dataset, num_splits: int, seed: int):
    """Shuffle and split a Dataset into `num_splits` contiguous parts."""
    print(f"\nShuffling and splitting data with seed {seed}...")
    shuffled = dataset.shuffle(seed=seed)

    total_rows = len(shuffled)
    split_size = total_rows // num_splits
    print(f"Total rows: {total_rows}. Each of {num_splits} splits will have ~{split_size} rows.")

    splits = []
    for i in range(num_splits):
        start = i * split_size
        end = (i + 1) * split_size if i < num_splits - 1 else total_rows  # last one takes remainder
        splits.append(shuffled.select(range(start, end)))
    return splits

def main():
    args = parse_args()

    # Ensure save directories exist
    ensure_dir(args.base_save_path)
    ensure_dir(args.json_save_path)

    # 1) Load dataset (all available splits)
    print(f"Loading the full dataset '{args.dataset_name}'...")
    full_ds = load_dataset(args.dataset_name)
    available_splits = list(full_ds.keys())
    print(f"Dataset loaded. Available splits: {available_splits}")

    # Guard: need at least one split
    if not available_splits:
        raise ValueError("No splits found in the dataset.")

    # Precompute split parts for each available split
    split_parts_by_name = {}  # e.g., {"train": [ds_part1, ds_part2, ...], "test": [...]}
    for split_name in available_splits:
        print(f"\nProcessing split: '{split_name}'")
        split_parts_by_name[split_name] = split_data(
            full_ds[split_name], args.num_splits, args.seed
        )

    # 2) Combine corresponding parts across splits and save
    for i in range(args.num_splits):
        part_num = i + 1
        print(f"\n--- Processing and saving Part {part_num} ---")

        # Build a DatasetDict with the i-th part for each split
        part_splits = {sn: split_parts_by_name[sn][i] for sn in available_splits}
        final_part = DatasetDict(part_splits)

        # Sizes printout
        for sn in available_splits:
            print(f"Part {part_num} '{sn}' size: {len(final_part[sn])}")

        # 2a) Save JSONL for each split
        for sn in available_splits:
            json_path = os.path.join(
                args.json_save_path, f"{args.dataset_name.replace('/', '_')}_part{part_num}_{sn}.jsonl"
            )
            final_part[sn].to_json(json_path)
            print(f"Saved JSONL for split '{sn}' to: {json_path}")

        # 2b) Save HF dataset to disk
        disk_dir = os.path.join(
            args.base_save_path, f"{args.dataset_name.replace('/', '_')}_part{part_num}"
        )
        final_part.save_to_disk(disk_dir)
        print(f"Saved dataset part {part_num} to disk at: {disk_dir}")

    print("\nAll parts processed and saved successfully!")

if __name__ == "__main__":
    main()
