import json
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

dataset_path = '/data/xiejingyi/dataset/yelp/yelp_academic_dataset_review.json'

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    exit(1)

print(f"Successfully loaded {len(data_list)} records")

if data_list:

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
            word_count = len(text.split())
            char_count = len(text)
            
            text_word_counts.append(word_count)
            text_char_counts.append(char_count)
    
    if text_word_counts:
        print(f"\nWord count statistics:")
        print(f"  Total samples: {len(text_word_counts)}")
        print(f"  Min words: {min(text_word_counts)}")
        print(f"  Max words: {max(text_word_counts)}")
        print(f"  Mean words: {np.mean(text_word_counts):.1f}")
        print(f"  Median words: {np.median(text_word_counts):.1f}")
        print(f"  Std dev: {np.std(text_word_counts):.1f}")
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nWord count percentiles:")
        for p in percentiles:
            value = np.percentile(text_word_counts, p)
            print(f"  {p}%: {value:.0f} words")
        
        print(f"\nCharacter count statistics:")
        print(f"  Mean chars: {np.mean(text_char_counts):.1f}")
        print(f"  Median chars: {np.median(text_char_counts):.1f}")
        print(f"  Mean chars per word: {np.mean(text_char_counts)/np.mean(text_word_counts):.1f}")
        
        print(f"\nWord count distribution:")
        word_bins = [0, 10, 20, 50, 100, 200, 500, float('inf')]
        word_bin_labels = ['0-10', '11-20', '21-50', '51-100', '101-200', '201-500', '500+']
        
        for i in range(len(word_bins)-1):
            if word_bins[i+1] == float('inf'):
                count = sum(1 for word_count in text_word_counts if word_count > word_bins[i])
            else:
                count = sum(1 for word_count in text_word_counts if word_bins[i] < word_count <= word_bins[i+1])
            percentage = (count / len(text_word_counts)) * 100
            print(f"  {word_bin_labels[i]}: {count} records ({percentage:.1f}%)")
        
    
    print(f"\n=== Dataset Processing Pipeline ===")
    
    if len(data_list) > 0:
        print(f"Step 1: Sample 200k records for train dataset")
        
        train_sample_size = 200000
        
        all_word_counts = []
        for record in data_list:
            if 'text' in record and record['text']:
                word_count = len(record['text'].split())
                all_word_counts.append(word_count)
        
        print(f"Original data size: {len(data_list):,} records")
        print(f"Valid text records: {len(all_word_counts):,}")
        
        if len(data_list) >= train_sample_size:
            print("Sampling train dataset [Random sampling]...")
            train_indices = random.sample(range(len(data_list)), train_sample_size)
            train_dataset = [data_list[i] for i in train_indices]
            print(f"Train dataset sampled: {len(train_dataset):,} records")
            
            print(f"\nStep 2: Create retain dataset (10th to 90th percentile by word count)")
            
            train_word_counts = []
            for record in train_dataset:
                if 'text' in record and record['text']:
                    word_count = len(record['text'].split())
                    train_word_counts.append(word_count)
            
            q10 = np.percentile(train_word_counts, 10)
            q90 = np.percentile(train_word_counts, 90)
            
            print(f"Train dataset word count distribution:")
            print(f"  10th percentile: {q10:.1f}")
            print(f"  90th percentile: {q90:.1f}")
            
            retain_dataset = []
            forget_dataset = []
            for record in train_dataset:
                if 'text' in record and record['text']:
                    word_count = len(record['text'].split())
                    if q10 <= word_count <= q90:
                        retain_dataset.append(record)
                    else:
                        forget_dataset.append(record)
                else:
                    forget_dataset.append(record)
            
            print(f"Retain dataset: {len(retain_dataset):,} records")
            print(f"Forget dataset: {len(forget_dataset):,} records")
            print(f"Removal ratio: {(len(train_dataset) - len(retain_dataset)) / len(train_dataset) * 100:.1f}%")
            
            
            output_dir = '/data/xiejingyi/dataset/yelp_10_90%'
            os.makedirs(output_dir, exist_ok=True)
            
            train_file = os.path.join(output_dir, 'yelp_train_200k.json')
            print(f"Saving train dataset...")
            with open(train_file, 'w', encoding='utf-8') as f:
                for record in tqdm(train_dataset, desc="Saving train", unit="records"):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Train dataset saved: {train_file}")
            
            retain_full_file = os.path.join(output_dir, 'yelp_retain_full.json')
            print(f"Saving retain full dataset...")
            with open(retain_full_file, 'w', encoding='utf-8') as f:
                for record in tqdm(retain_dataset, desc="Saving retain full", unit="records"):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Retain full dataset saved: {retain_full_file}")
            
            forget_file = os.path.join(output_dir, 'yelp_forget.json')
            print(f"Saving forget dataset...")
            with open(forget_file, 'w', encoding='utf-8') as f:
                for record in tqdm(forget_dataset, desc="Saving forget", unit="records"):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Forget dataset saved: {forget_file}")
            
            dataset_info = {
                'random_seed': RANDOM_SEED,
                'original_size': len(data_list),
                'train_size': len(train_dataset),
                'retain_size': len(retain_dataset),
                'forget_size': len(forget_dataset),
                'word_count_quartiles': {
                    'q_lower': float(q10),
                    'q_upper': float(q90),
                },
                'train_word_stats': {
                    'mean': float(np.mean(train_word_counts)),
                    'median': float(np.median(train_word_counts)),
                    'std': float(np.std(train_word_counts))
                },
            }
            
            info_file = os.path.join(output_dir, 'yelp_dataset_info.json')
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            print(f"Dataset info saved: {info_file}")
            
            print(f"\nStep 4: Sample additional 200k records for vocabulary building")
            
            vocab_sample_size = 200000
            
            print("Sampling vocabulary dataset [Random sampling]...")
            # Exclude indices already used in train_dataset to avoid overlap
            train_indices_set = set(train_indices)
            available_indices = [i for i in range(len(data_list)) if i not in train_indices_set]
            
            if len(available_indices) >= vocab_sample_size:
                vocab_indices = random.sample(available_indices, vocab_sample_size)
            else:
                # If not enough available, sample from all data (with possible overlap)
                vocab_indices = random.sample(range(len(data_list)), vocab_sample_size)
            
            vocab_dataset = [data_list[i] for i in vocab_indices]
            print(f"Vocabulary dataset sampled: {len(vocab_dataset):,} records")
            
            vocab_file = os.path.join(output_dir, 'yelp_vocab_200k.json')
            print(f"Saving vocabulary dataset...")
            with open(vocab_file, 'w', encoding='utf-8') as f:
                for record in tqdm(vocab_dataset, desc="Saving vocab dataset", unit="records"):
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Vocabulary dataset saved: {vocab_file}")
            
        else:
            print(f"Original data size insufficient for {train_sample_size:,} records")
            
    else:
        print("No text field found or text field is empty")

else:
    print("No data loaded successfully")
