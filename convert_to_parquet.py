#!/usr/bin/env python3
"""
Convert JSONL Q&A data to Parquet format for efficient ML training
"""
import json
import pandas as pd
from pathlib import Path

def convert_jsonl_to_parquet():
    # Input and output paths
    input_file = "data/final_qa_data_unique.jsonl"
    output_file = "data/qa_dataset.parquet"

    print(f"Converting {input_file} to {output_file}...")

    # Read JSONL file
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                # Create the concatenated full-question field
                full_question = f"For {record['topic']}, {record['question']}"

                # Create new record with only the two required fields
                new_record = {
                    "full-question": full_question,
                    "answer": record['answer']
                }
                data.append(new_record)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping invalid record on line {line_num}: {e}")

    print(f"Loaded {len(data)} records from JSONL")

    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Display sample data
    print("\nSample records:")
    for i in range(min(3, len(df))):
        print(f"\nRecord {i+1}:")
        print(f"Full-question: {df.iloc[i]['full-question']}")
        print(f"Answer: {df.iloc[i]['answer']}")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values per column:")
        print(missing[missing > 0])

    # Save as Parquet
    df.to_parquet(output_file, index=False, compression='snappy')

    # Compare file sizes
    input_size = Path(input_file).stat().st_size / 1024 / 1024  # MB
    output_size = Path(output_file).stat().st_size / 1024 / 1024  # MB

    print(f"\nConversion complete!")
    print(f"Original JSONL: {input_size:.2f} MB")
    print(f"Parquet file: {output_size:.2f} MB")
    print(f"Size reduction: {((input_size - output_size) / input_size * 100):.1f}%")

    return df

if __name__ == "__main__":
    df = convert_jsonl_to_parquet()