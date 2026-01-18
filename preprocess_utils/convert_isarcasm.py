"""
Script to convert iSarcasm dataset from HuggingFace to DRAGON's expected format
For BINARY CLASSIFICATION (not QA-style multiple choice)

USAGE:
 python preprocess_utils/convert_isarcasm.py

Input: HuggingFace dataset "viethq1906/isarcasm_2022_taskA_En"
Output: JSONL files in DRAGON format for binary classification
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm

__all__ = ['convert_isarcasm_to_dragon']


def convert_isarcasm_to_dragon(output_dir: str = './data/isarcasm'):
    """
    Convert iSarcasm dataset to DRAGON's expected format for BINARY classification.
    
    Binary labels: 0 (non-sarcastic), 1 (sarcastic)
    Format follows CSQA but adapted for single-sentence classification.
    """
    print(f'Loading iSarcasm dataset from HuggingFace...')
    
    # Load the dataset
    dataset = load_dataset("viethq1906/isarcasm_2022_taskA_En")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'statement'), exist_ok=True)
    
    # Process each split
    splits = {
        'train': 'train',
        'validation': 'dev',
        'test': 'test'
    }
    
    for hf_split, dragon_split in splits.items():
        if hf_split not in dataset:
            print(f"Split '{hf_split}' not found in dataset, skipping...")
            continue
            
        split_data = dataset[hf_split]
        output_file = os.path.join(output_dir, f'{dragon_split}.jsonl')
        
        print(f'Converting {hf_split} split to {output_file}...')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(tqdm(split_data, desc=f'Processing {hf_split}')):
                # Get the text and label
                text = example.get('text', example.get('tweet', example.get('sentence', '')))
                label = example.get('label', example.get('sarcastic', 0))
                
                # Create unique ID
                example_id = f"isarcasm_{hf_split}_{idx}"
                
                # BINARY CLASSIFICATION FORMAT
                # We create a single statement for grounding (to extract concepts)
                # The statement is just the text itself
                dragon_example = {
                    "id": example_id,
                    "question": {
                        "stem": text,  # The text to classify
                        "choices": []   # Empty for binary classification
                    },
                    "answerKey": str(label),  # "0" or "1"
                    "statements": [
                        {
                            "label": True,  # We only use this for concept extraction
                            "statement": text  # Just the text, no appended choice
                        }
                    ]
                }
                
                f.write(json.dumps(dragon_example, ensure_ascii=False))
                f.write('\n')
        
        print(f'Converted {len(split_data)} examples to {output_file}')
    
    print(f'\nDataset conversion complete!')
    print(f'Files saved to: {output_dir}/')
    print(f'\nNext steps:')
    print(f'1. Run grounding: python preprocess.py --run isarcasm_ground')
    print(f'2. Run graph extraction: python preprocess.py --run isarcasm_graph')


if __name__ == '__main__':
    convert_isarcasm_to_dragon()
