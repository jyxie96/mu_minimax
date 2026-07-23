import argparse
import json
import math
import os
import random
import sys

import numpy as np
from tqdm import tqdm

dataset_path = '/data/xiejingyi/dataset/yelp/yelp_academic_dataset_review.json'
OUTPUT_DIR = '/data/xiejingyi/dataset/yelp_longest_5pct'
POOL_N = 200_000
FORGET_RATIO = 0.05

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def save_jsonl(records, output_path, desc):
    print(f"Saving {desc}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in tqdm(records, desc=desc, unit="records"):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"{desc} saved: {output_path}")


def subsample_records(records, n, seed=RANDOM_SEED):
    n = min(int(n), len(records))
    rng = random.Random(seed)
    return rng.sample(records, n)


def split_forget_by_char_length(records, forget_ratio=FORGET_RATIO):
    """Within a record pool, take the longest forget_ratio fraction as forget."""
    char_lengths = [len(rec.get('text') or '') for rec in records]
    n_pool = len(records)
    n_forget = max(1, int(math.ceil(n_pool * forget_ratio)))

    sorted_idx = np.argsort(np.asarray(char_lengths), kind='stable')
    forget_idx = set(sorted_idx[-n_forget:].tolist())

    forget_dataset = [rec for i, rec in enumerate(records) if i in forget_idx]
    retain_dataset = [rec for i, rec in enumerate(records) if i not in forget_idx]
    return retain_dataset, forget_dataset


def reservoir_sample_jsonl_records(input_path, n, seed=RANDOM_SEED):
    """Reservoir-sample n parsed JSON records from a JSONL file."""
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0

    print(f"Reservoir-sampling {n:,} records from {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSONL", unit="lines"):
            if not line.strip():
                continue
            record = json.loads(line)
            if len(reservoir) < n:
                reservoir.append(record)
            else:
                j = rng.randint(0, n_seen)
                if j < n:
                    reservoir[j] = record
            n_seen += 1

    if n_seen == 0:
        raise ValueError(f"No records found in {input_path}")

    if n_seen < n:
        print(f"Warning: only {n_seen:,} records available; using all of them.")
        return reservoir[:n_seen], n_seen

    print(f"Sampled {len(reservoir):,} records from {n_seen:,} total records.")
    return reservoir, n_seen


def build_pool_retain_forget_split(
    pool_records,
    output_dir=OUTPUT_DIR,
    pool_n=POOL_N,
    forget_ratio=FORGET_RATIO,
    pool_seed=RANDOM_SEED,
    source_size=None,
):
    """Save 200k pool, then retain/forget split where forget = longest 5% within pool."""
    os.makedirs(output_dir, exist_ok=True)

    if len(pool_records) > pool_n:
        pool_records = subsample_records(pool_records, pool_n, seed=pool_seed)
    pool_size = len(pool_records)

    retain_dataset, forget_dataset = split_forget_by_char_length(pool_records, forget_ratio=forget_ratio)

    pool_file = os.path.join(output_dir, 'yelp_pool_200k.json')
    retain_file = os.path.join(output_dir, 'yelp_retain.json')
    forget_file = os.path.join(output_dir, 'yelp_forget.json')

    save_jsonl(pool_records, pool_file, "200k random pool")
    save_jsonl(retain_dataset, retain_file, "retain dataset")
    save_jsonl(forget_dataset, forget_file, "forget dataset")

    pool_char_lengths = [len(rec.get('text') or '') for rec in pool_records]
    info = {
        'random_seed': RANDOM_SEED,
        'protocol': 'random_sample_pool_then_longest_5pct_forget',
        'source_size': source_size,
        'pool_n': int(pool_n),
        'pool_sample_seed': int(pool_seed),
        'pool_size': pool_size,
        'pool_file': os.path.basename(pool_file),
        'forget_ratio_within_pool': float(forget_ratio),
        'forget_size': len(forget_dataset),
        'retain_size': len(retain_dataset),
        'retain_file': os.path.basename(retain_file),
        'forget_file': os.path.basename(forget_file),
        'pool_text_char_length_stats': {
            'mean': float(np.mean(pool_char_lengths)),
            'median': float(np.median(pool_char_lengths)),
            'std': float(np.std(pool_char_lengths)),
        },
    }

    info_path = os.path.join(output_dir, 'yelp_dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Dataset info saved: {info_path}")
    print(
        f"Pool={pool_size:,}, retain={len(retain_dataset):,}, "
        f"forget={len(forget_dataset):,} ({len(forget_dataset)/pool_size*100:.1f}% of pool)"
    )


def build_200k_split_from_train(
    train_path=None,
    output_dir=OUTPUT_DIR,
    pool_n=POOL_N,
    forget_ratio=FORGET_RATIO,
    pool_seed=RANDOM_SEED,
):
    train_path = train_path or os.path.join(output_dir, 'yelp_train_full.json')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    pool_records, source_size = reservoir_sample_jsonl_records(train_path, pool_n, seed=pool_seed)
    build_pool_retain_forget_split(
        pool_records,
        output_dir=output_dir,
        pool_n=pool_n,
        forget_ratio=forget_ratio,
        pool_seed=pool_seed,
        source_size=source_size,
    )


def run_full_pipeline():
    data_list = []

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        data_list.append(data)
                        if len(data_list) % 10000 == 0:
                            print(f"Loaded {len(data_list)} records...")
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error at line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        sys.exit(1)

    print(f"Successfully loaded {len(data_list)} records")

    if not data_list:
        print("No data loaded successfully")
        return

    print(f"\n=== Sample Data ===")
    for i, record in enumerate(data_list[:3]):
        print(f"\nRecord {i+1}:")
        for key, value in record.items():
            if isinstance(value, str) and len(value) > 100:
                display_value = value[:100] + "..."
            else:
                display_value = value
            print(f"  {key}: {display_value}")

    print(f"\n=== Text Field Word Count Distribution ===")

    text_word_counts = []
    text_char_counts = []

    for record in data_list:
        if 'text' in record and record['text']:
            text = record['text']
            text_word_counts.append(len(text.split()))
            text_char_counts.append(len(text))

    if text_word_counts:
        print(f"\nWord count statistics:")
        print(f"  Total samples: {len(text_word_counts)}")
        print(f"  Min words: {min(text_word_counts)}")
        print(f"  Max words: {max(text_word_counts)}")
        print(f"  Mean words: {np.mean(text_word_counts):.1f}")
        print(f"  Median words: {np.median(text_word_counts):.1f}")
        print(f"  Std dev: {np.std(text_word_counts):.1f}")

    print(f"\n=== Dataset Processing Pipeline ===")
    print("Step 1: Use full dataset for training set (no sampling)")

    train_dataset = data_list
    print(f"Train dataset size: {len(train_dataset):,} records")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_file = os.path.join(OUTPUT_DIR, 'yelp_train_full.json')
    save_jsonl(train_dataset, train_file, "train dataset")

    print(f"\nStep 2: Random sample {POOL_N:,} records from full train")
    pool_records = subsample_records(train_dataset, POOL_N, seed=RANDOM_SEED)

    print(f"\nStep 3: Within the 200k pool, take longest {FORGET_RATIO*100:.0f}% as forget")
    build_pool_retain_forget_split(
        pool_records,
        output_dir=OUTPUT_DIR,
        pool_n=POOL_N,
        forget_ratio=FORGET_RATIO,
        pool_seed=RANDOM_SEED,
        source_size=len(train_dataset),
    )

    print("\nStep 4: Use full dataset for vocabulary building (no sampling)")
    vocab_file = os.path.join(OUTPUT_DIR, 'yelp_vocab_full.json')
    save_jsonl(train_dataset, vocab_file, "vocabulary dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Yelp retain/forget datasets.")
    parser.add_argument(
        '--build-200k-split-only',
        action='store_true',
        help='Sample 200k from yelp_train_full.json, then split longest 5%% into forget.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.build_200k_split_only:
        build_200k_split_from_train()
    else:
        run_full_pipeline()
