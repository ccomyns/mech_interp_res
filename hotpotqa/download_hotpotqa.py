#!/usr/bin/env python3
"""
Download HotpotQA dataset from HuggingFace and save to local JSON.

The HotpotQA dataset (distractor setting) contains:
- Supporting paragraphs that contain the answer
- Distractor paragraphs that are topically related but don't contain the answer

This is useful for testing if a linear probe generalizes to detecting
when a model might be misled by irrelevant/distracting information.

Usage:
    python download_hotpotqa.py --test        # Download a few entries to inspect structure
    python download_hotpotqa.py               # Download full dataset
    python download_hotpotqa.py --split train # Download specific split (train/validation)
    python download_hotpotqa.py --balanced 1000  # Download 1000 of each difficulty level
    python download_hotpotqa.py --balanced 1000 --ascii-only  # Only ASCII characters
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def contains_non_ascii(text: str) -> bool:
    """Check if text contains any non-ASCII characters."""
    return any(ord(char) > 127 for char in text)


def entry_has_non_ascii(example: dict) -> bool:
    """
    Check if a HotpotQA example contains non-ASCII characters in key fields.
    
    Checks: question, answer, and all context sentences/titles.
    """
    # Check question and answer
    if contains_non_ascii(example.get('question', '')):
        return True
    if contains_non_ascii(example.get('answer', '')):
        return True
    
    # Check context titles and sentences
    context = example.get('context', {})
    for title in context.get('title', []):
        if contains_non_ascii(title):
            return True
    for sentences in context.get('sentences', []):
        for sentence in sentences:
            if contains_non_ascii(sentence):
                return True
    
    return False


def download_hotpotqa(test_mode: bool = False, split: str = "validation", num_test_samples: int = 5,
                      balanced_per_level: int = None, ascii_only: bool = False):
    """
    Download HotpotQA dataset from HuggingFace.
    
    Args:
        test_mode: If True, only download a few samples for inspection
        split: Which split to download ('train' or 'validation')
        num_test_samples: Number of samples to download in test mode
        balanced_per_level: If set, download this many samples per difficulty level
        ascii_only: If True, filter out entries with non-ASCII characters
    
    Returns:
        The dataset (or subset in test mode)
    """
    from datasets import load_dataset
    
    print(f"Loading HotpotQA dataset (distractor setting), split: {split}...")
    if ascii_only:
        print("[ASCII-ONLY MODE] Will filter out entries with non-ASCII characters")
    
    # Load the distractor setting - this includes misleading paragraphs
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    
    if test_mode:
        print(f"\n[TEST MODE] Selecting {num_test_samples} samples for inspection...")
        dataset = dataset.select(range(min(num_test_samples, len(dataset))))
    elif balanced_per_level is not None:
        print(f"\n[BALANCED MODE] Selecting {balanced_per_level} samples per difficulty level...")
        dataset = select_balanced_by_level(dataset, balanced_per_level, ascii_only=ascii_only)
    
    print(f"Loaded {len(dataset)} examples")
    return dataset


def select_balanced_by_level(dataset, num_per_level: int, ascii_only: bool = False):
    """
    Select a balanced number of samples from each difficulty level.
    
    Args:
        dataset: The full HotpotQA dataset
        num_per_level: Number of samples to select per difficulty level
        ascii_only: If True, skip entries with non-ASCII characters
    
    Returns:
        A filtered dataset with balanced difficulty levels
    """
    # Group indices by difficulty level
    level_indices = {'easy': [], 'medium': [], 'hard': []}
    skipped_non_ascii = {'easy': 0, 'medium': 0, 'hard': 0}
    
    for idx, item in enumerate(dataset):
        level = item.get('level', 'unknown')
        if level in level_indices:
            # Check for non-ASCII if filtering is enabled
            if ascii_only and entry_has_non_ascii(item):
                skipped_non_ascii[level] += 1
                continue
            level_indices[level].append(idx)
    
    # Report available counts
    print("\nAvailable examples per level (after filtering):")
    for level, indices in level_indices.items():
        skipped = skipped_non_ascii[level]
        if ascii_only and skipped > 0:
            print(f"  {level}: {len(indices)} (skipped {skipped} with non-ASCII)")
        else:
            print(f"  {level}: {len(indices)}")
    
    # Select samples from each level
    selected_indices = []
    for level in ['easy', 'medium', 'hard']:
        indices = level_indices[level]
        num_to_select = min(num_per_level, len(indices))
        if num_to_select < num_per_level:
            print(f"  Warning: Only {num_to_select} {level} examples available (requested {num_per_level})")
        selected_indices.extend(indices[:num_to_select])
    
    print(f"\nSelected {len(selected_indices)} total examples")
    
    # Return filtered dataset
    return dataset.select(selected_indices)


def inspect_structure(dataset):
    """Print the structure of the dataset for inspection."""
    print("\n" + "="*80)
    print("DATASET STRUCTURE INSPECTION")
    print("="*80)
    
    # Get first example
    example = dataset[0]
    
    print("\n📋 FIELDS IN EACH EXAMPLE:")
    for key in example.keys():
        value = example[key]
        if isinstance(value, list):
            print(f"  - {key}: list of {len(value)} items")
            if len(value) > 0:
                print(f"      First item type: {type(value[0]).__name__}")
                if isinstance(value[0], str) and len(value[0]) > 100:
                    print(f"      First item preview: {value[0][:100]}...")
                elif isinstance(value[0], list):
                    print(f"      First item: {value[0][:3]}..." if len(value[0]) > 3 else f"      First item: {value[0]}")
                else:
                    print(f"      First item: {value[0]}")
        elif isinstance(value, str):
            if len(value) > 100:
                print(f"  - {key}: str, preview: {value[:100]}...")
            else:
                print(f"  - {key}: str = '{value}'")
        else:
            print(f"  - {key}: {type(value).__name__} = {value}")
    
    print("\n" + "-"*80)
    print("📖 FULL FIRST EXAMPLE:")
    print("-"*80)
    print(json.dumps(example, indent=2, default=str))
    
    print("\n" + "-"*80)
    print("📊 CONTEXT STRUCTURE BREAKDOWN:")
    print("-"*80)
    
    # The context field contains titles and sentences
    if 'context' in example:
        context = example['context']
        titles = context.get('title', [])
        sentences = context.get('sentences', [])
        
        print(f"\nNumber of paragraphs: {len(titles)}")
        print("\nParagraph details:")
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            is_supporting = title in example.get('supporting_facts', {}).get('title', [])
            marker = "✓ SUPPORTING" if is_supporting else "✗ DISTRACTOR"
            print(f"\n  [{i}] {marker}: {title}")
            print(f"      Sentences: {len(sents)}")
            preview = ' '.join(sents)[:150] + "..." if len(' '.join(sents)) > 150 else ' '.join(sents)
            print(f"      Preview: {preview}")
    
    print("\n" + "-"*80)
    print("❓ QUESTION AND ANSWER:")
    print("-"*80)
    print(f"\nQuestion: {example.get('question', 'N/A')}")
    print(f"Answer: {example.get('answer', 'N/A')}")
    print(f"Type: {example.get('type', 'N/A')}")
    print(f"Level: {example.get('level', 'N/A')}")


def convert_to_local_format(dataset):
    """
    Convert HotpotQA dataset to a local JSON format.
    
    The format is designed to be similar to SQuAD for consistency,
    but includes metadata about which paragraphs are supporting vs distractors.
    """
    converted_data = []
    
    for idx, example in enumerate(dataset):
        # Get supporting paragraph titles
        supporting_titles = set(example['supporting_facts']['title'])
        
        # Build context with paragraph metadata
        paragraphs_info = []
        full_context_parts = []
        
        for title, sentences in zip(example['context']['title'], example['context']['sentences']):
            paragraph_text = ' '.join(sentences)
            is_supporting = title in supporting_titles
            
            paragraphs_info.append({
                'title': title,
                'text': paragraph_text,
                'is_supporting': is_supporting,
                'is_distractor': not is_supporting
            })
            
            full_context_parts.append(f"[{title}]: {paragraph_text}")
        
        # Create the converted entry
        entry = {
            'id': example['id'],
            'question': example['question'],
            'answer': example['answer'],
            'type': example['type'],  # 'bridge' or 'comparison'
            'level': example['level'],  # 'easy', 'medium', 'hard'
            'full_context': '\n\n'.join(full_context_parts),
            'paragraphs': paragraphs_info,
            'supporting_facts': {
                'titles': list(supporting_titles),
                'sentence_indices': list(zip(
                    example['supporting_facts']['title'],
                    example['supporting_facts']['sent_id']
                ))
            },
            'num_supporting': sum(1 for p in paragraphs_info if p['is_supporting']),
            'num_distractors': sum(1 for p in paragraphs_info if p['is_distractor'])
        }
        
        converted_data.append(entry)
    
    return converted_data


def save_to_json(data, output_path: Path, test_mode: bool = False):
    """Save the converted data to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(data)} examples to: {output_path}")
    
    if test_mode:
        print("\n📝 SAMPLE OUTPUT STRUCTURE:")
        print("-"*80)
        print(json.dumps(data[0], indent=2)[:2000])
        if len(json.dumps(data[0], indent=2)) > 2000:
            print("... [truncated]")


def main():
    parser = argparse.ArgumentParser(
        description="Download HotpotQA dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_hotpotqa.py --test              # Inspect structure with 5 samples
  python download_hotpotqa.py --test -n 10        # Inspect with 10 samples
  python download_hotpotqa.py --split validation  # Download validation set
  python download_hotpotqa.py --split train       # Download training set
  python download_hotpotqa.py --balanced 1000     # Download 1000 easy + 1000 medium + 1000 hard
  python download_hotpotqa.py --balanced 1000 --ascii-only  # Same but only ASCII characters
        """
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test mode: download only a few samples to inspect the structure'
    )
    
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=5,
        help='Number of samples to download in test mode (default: 5)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='validation',
        choices=['train', 'validation'],
        help='Which split to download (default: validation)'
    )
    
    parser.add_argument(
        '--balanced',
        type=int,
        default=None,
        metavar='N',
        help='Download N samples per difficulty level (easy, medium, hard). Requires --split train.'
    )
    
    parser.add_argument(
        '--ascii-only',
        action='store_true',
        help='Filter out entries containing non-ASCII unicode characters (accents, Cyrillic, etc.)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: hotpotqa_{split}[_test].json)'
    )
    
    args = parser.parse_args()
    
    # Validate balanced mode requires training split
    if args.balanced is not None and args.split != 'train':
        print("Warning: --balanced requires --split train (validation set only has 'hard' examples)")
        print("Automatically switching to training split...")
        args.split = 'train'
    
    # Download dataset
    dataset = download_hotpotqa(
        test_mode=args.test,
        split=args.split,
        num_test_samples=args.num_samples,
        balanced_per_level=args.balanced,
        ascii_only=args.ascii_only
    )
    
    # In test mode, inspect the structure first
    if args.test:
        inspect_structure(dataset)
    
    # Convert to local format
    print("\nConverting to local JSON format...")
    converted_data = convert_to_local_format(dataset)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.test:
            suffix = "_test"
        elif args.balanced:
            suffix = f"_balanced_{args.balanced}each"
            if args.ascii_only:
                suffix += "_asciionly"
        else:
            suffix = "_asciionly" if args.ascii_only else ""
        output_path = Path(f"hotpotqa_{args.split}{suffix}_{timestamp}.json")
    
    # Save to JSON
    save_to_json(converted_data, output_path, test_mode=args.test)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. If you ran with --test, inspect the output JSON to understand the structure.

2. The 'paragraphs' field contains both supporting and distractor paragraphs.
   - is_supporting=True: paragraphs that contain relevant info for answering
   - is_distractor=True: paragraphs that are misleading/irrelevant

3. For your linear probe experiment, you could:
   - Use 'full_context' as the context (contains both supporting + distractors)
   - Compare model behavior on supporting-only vs full context
   - The distractors serve a similar role to your "incorrect hints"

4. To download the full dataset, run without --test:
   python download_hotpotqa.py --split validation
   python download_hotpotqa.py --split train
    """)


if __name__ == "__main__":
    main()

