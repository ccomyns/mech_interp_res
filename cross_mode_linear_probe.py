#!/usr/bin/env python3
"""
Cross-Mode Linear Probe Evaluation

This script evaluates a linear probe trained on one incorrect hint mode
by testing it on data from a different incorrect hint mode.

This tests whether the probe's learned representation of "correct answer vs
agrees with hint" generalizes across different hint formats.

Usage:
    python cross_mode_linear_probe.py <probe_json> <categorized_json> <bulk_json> [options]

Example:
    python cross_mode_linear_probe.py \
        probe_results/probe_correct_vs_hint_layer18_manual_review.json \
        linear_probe_mode_2/categorized_data_no_math_20251217_150300.json \
        linear_probe_mode_2/bulk_residual_stream_incorrect_hint_2_q0-1139_tokens1_20251216_224948.json
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from datetime import datetime
from pathlib import Path


def load_probe(probe_path: str) -> Tuple[np.ndarray, float, int]:
    """
    Load a trained linear probe from JSON.
    
    Args:
        probe_path: Path to probe JSON file
        
    Returns:
        Tuple of (weights array, bias, layer number)
    """
    print(f"\n📂 Loading probe from: {probe_path}")
    
    with open(probe_path, 'r', encoding='utf-8') as f:
        probe_data = json.load(f)
    
    weights = np.array(probe_data['probe_weights'])
    bias = probe_data['probe_bias']
    layer = probe_data.get('layer', 18)
    
    print(f"✅ Loaded probe")
    print(f"   Layer: {layer}")
    print(f"   Feature dimension: {len(weights)}")
    print(f"   Bias: {bias:.6f}")
    
    return weights, bias, layer


def load_categorized_data(json_path: str) -> Dict[str, List[Dict]]:
    """
    Load categorized data from JSON file.
    
    Returns:
        Dictionary with 'correct_answer' and 'agrees_with_hint' lists
    """
    print(f"\n📂 Loading categorized data from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categorized_data = {
        'correct_answer': data.get('correct_answer', []),
        'agrees_with_hint': data.get('agrees_with_hint', [])
    }
    
    print(f"✅ Loaded categorized data")
    print(f"   correct_answer: {len(categorized_data['correct_answer'])} questions")
    print(f"   agrees_with_hint: {len(categorized_data['agrees_with_hint'])} questions")
    
    return categorized_data


def load_residual_streams(bulk_path: str, question_indices: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Load residual stream data for specific question indices.
    
    Args:
        bulk_path: Path to bulk residual stream JSON file
        question_indices: List of question indices to load
        
    Returns:
        Dictionary mapping question_idx to residual stream data
    """
    print(f"\n📂 Loading residual streams from: {bulk_path}")
    print(f"   Looking for {len(question_indices)} questions...")
    
    target_indices = set(question_indices)
    residual_streams = {}
    
    with open(bulk_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    for result in results:
        q_idx = result.get('question_idx')
        if q_idx in target_indices:
            residual_streams[q_idx] = {
                'residual_stream_hidden_states': result.get('residual_stream_hidden_states', {}),
                'question_idx': q_idx
            }
    
    print(f"✅ Loaded residual streams for {len(residual_streams)}/{len(target_indices)} questions")
    
    missing = target_indices - set(residual_streams.keys())
    if missing:
        print(f"⚠️  Warning: {len(missing)} questions not found in bulk file")
    
    return residual_streams


def extract_features(categorized_data: Dict[str, List[Dict]],
                     residual_streams: Dict[int, Dict[str, Any]],
                     layer: int,
                     split: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Extract feature vectors and labels from categorized data.
    
    Args:
        categorized_data: Dictionary with 'correct_answer' and 'agrees_with_hint' lists
        residual_streams: Dictionary mapping question_idx to residual stream data
        layer: Which layer to extract
        split: If specified, only extract this split ('train' or 'test'). If None, extract all.
        
    Returns:
        Tuple of (features array, labels array, question indices, category names)
    """
    features = []
    labels = []
    question_indices = []
    categories = []
    
    layer_key = f"layer_{layer}"
    
    # Process correct_answer category (label = -1)
    for item in categorized_data['correct_answer']:
        if split is not None and item.get('split') != split:
            continue
            
        q_idx = item['question_idx']
        
        if q_idx not in residual_streams:
            continue
        
        hidden_states = residual_streams[q_idx].get('residual_stream_hidden_states', {})
        
        if layer_key not in hidden_states:
            continue
        
        layer_data = hidden_states[layer_key]
        if not layer_data:
            continue
        
        # Get the single position available
        position_key = list(layer_data.keys())[0]
        hidden_vector = np.array(layer_data[position_key])
        
        features.append(hidden_vector)
        labels.append(-1)
        question_indices.append(q_idx)
        categories.append('correct_answer')
    
    # Process agrees_with_hint category (label = +1)
    for item in categorized_data['agrees_with_hint']:
        if split is not None and item.get('split') != split:
            continue
            
        q_idx = item['question_idx']
        
        if q_idx not in residual_streams:
            continue
        
        hidden_states = residual_streams[q_idx].get('residual_stream_hidden_states', {})
        
        if layer_key not in hidden_states:
            continue
        
        layer_data = hidden_states[layer_key]
        if not layer_data:
            continue
        
        position_key = list(layer_data.keys())[0]
        hidden_vector = np.array(layer_data[position_key])
        
        features.append(hidden_vector)
        labels.append(+1)
        question_indices.append(q_idx)
        categories.append('agrees_with_hint')
    
    return np.array(features), np.array(labels), question_indices, categories


def apply_probe(features: np.ndarray, weights: np.ndarray, bias: float,
                scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the linear probe to features to get predictions.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        weights: Probe weight vector
        bias: Probe bias
        scale_features: Whether to standardize features before applying probe
        
    Returns:
        Tuple of (predicted labels, raw scores)
    """
    if scale_features:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features
    
    # Compute raw scores: w · x + b
    scores = features_scaled @ weights + bias
    
    # Convert to predictions: positive score -> +1, negative score -> -1
    predictions = np.sign(scores)
    predictions[predictions == 0] = 1  # Handle edge case
    
    return predictions.astype(int), scores


def print_results(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray,
                  probe_source: str, eval_target: str):
    """Print formatted evaluation results."""
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*80}")
    print(f"CROSS-MODE LINEAR PROBE EVALUATION")
    print(f"{'='*80}")
    print(f"\n📊 Setup:")
    print(f"   Probe trained on: {probe_source}")
    print(f"   Evaluated on: {eval_target}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    
    print(f"\n📋 Class Distribution:")
    print(f"   Correct Answer (-1): {sum(y_true == -1)} samples")
    print(f"   Agrees with Hint (+1): {sum(y_true == 1)} samples")
    
    print(f"\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, 
                                   target_names=['Correct Answer (-1)', 'Agrees with Hint (+1)'])
    print(report)
    
    print(f"📊 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"   Predicted:  -1    +1")
    print(f"   Actual -1: {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Actual +1: {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Score distribution analysis
    print(f"\n📊 Score Distribution:")
    correct_scores = scores[y_true == -1]
    hint_scores = scores[y_true == 1]
    
    print(f"   Correct Answer (-1) scores:")
    print(f"      Mean: {correct_scores.mean():.4f}, Std: {correct_scores.std():.4f}")
    print(f"      Min: {correct_scores.min():.4f}, Max: {correct_scores.max():.4f}")
    
    print(f"   Agrees with Hint (+1) scores:")
    print(f"      Mean: {hint_scores.mean():.4f}, Std: {hint_scores.std():.4f}")
    print(f"      Min: {hint_scores.min():.4f}, Max: {hint_scores.max():.4f}")
    
    # Score separation
    separation = hint_scores.mean() - correct_scores.mean()
    print(f"\n   Score separation (hint_mean - correct_mean): {separation:.4f}")
    
    print(f"{'='*80}")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': cm.tolist(),
        'correct_answer_scores': {
            'mean': float(correct_scores.mean()),
            'std': float(correct_scores.std()),
            'min': float(correct_scores.min()),
            'max': float(correct_scores.max())
        },
        'agrees_with_hint_scores': {
            'mean': float(hint_scores.mean()),
            'std': float(hint_scores.std()),
            'min': float(hint_scores.min()),
            'max': float(hint_scores.max())
        },
        'score_separation': float(separation)
    }


def save_results(results: Dict, output_path: str, config: Dict):
    """Save evaluation results to JSON."""
    output_data = {
        'metadata': {
            'evaluation_date': datetime.now().isoformat(),
            'probe_source': config['probe_path'],
            'eval_categorized': config['categorized_path'],
            'eval_bulk': config['bulk_path'],
            'layer': config['layer'],
            'split_evaluated': config.get('split', 'all')
        },
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved results to: {output_path}")


def run_cross_mode_evaluation(probe_path: str,
                               categorized_path: str,
                               bulk_path: str,
                               split: Optional[str] = None,
                               output_path: Optional[str] = None):
    """
    Run cross-mode linear probe evaluation.
    
    Args:
        probe_path: Path to trained probe JSON
        categorized_path: Path to categorized data JSON for evaluation
        bulk_path: Path to bulk residual stream JSON for evaluation
        split: Optional split to evaluate ('train', 'test', or None for all)
        output_path: Optional path to save results
    """
    # Load probe
    weights, bias, layer = load_probe(probe_path)
    
    # Load categorized data
    categorized_data = load_categorized_data(categorized_path)
    
    # Get all question indices
    all_indices = []
    for item in categorized_data['correct_answer']:
        if split is None or item.get('split') == split:
            all_indices.append(item['question_idx'])
    for item in categorized_data['agrees_with_hint']:
        if split is None or item.get('split') == split:
            all_indices.append(item['question_idx'])
    
    # Load residual streams
    residual_streams = load_residual_streams(bulk_path, all_indices)
    
    # Extract features
    split_desc = split if split else "all"
    print(f"\n🔄 Extracting features (split: {split_desc}) from layer {layer}...")
    
    X, y_true, q_indices, categories = extract_features(
        categorized_data, residual_streams, layer, split
    )
    
    if len(X) == 0:
        raise ValueError("No valid features extracted!")
    
    print(f"✅ Extracted {len(X)} feature vectors")
    print(f"   Correct Answer (-1): {sum(y_true == -1)}")
    print(f"   Agrees with Hint (+1): {sum(y_true == 1)}")
    
    # Apply probe
    print(f"\n🔮 Applying probe to features...")
    y_pred, scores = apply_probe(X, weights, bias)
    
    # Print results
    probe_name = Path(probe_path).stem
    eval_name = Path(categorized_path).stem
    results = print_results(y_true, y_pred, scores, probe_name, eval_name)
    
    # Save results if requested
    if output_path:
        config = {
            'probe_path': probe_path,
            'categorized_path': categorized_path,
            'bulk_path': bulk_path,
            'layer': layer,
            'split': split
        }
        save_results(results, output_path, config)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a linear probe trained on one mode using data from another mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate mode 1 probe on mode 2 data (all splits)
    python cross_mode_linear_probe.py \\
        probe_results/probe_correct_vs_hint_layer18_manual_review.json \\
        linear_probe_mode_2/categorized_data_no_math_20251217_150300.json \\
        linear_probe_mode_2/bulk_residual_stream_incorrect_hint_2_q0-1139_tokens1_20251216_224948.json

    # Evaluate only on test split
    python cross_mode_linear_probe.py \\
        probe_results/probe_correct_vs_hint_layer18_manual_review.json \\
        linear_probe_mode_2/categorized_data_no_math_20251217_150300.json \\
        linear_probe_mode_2/bulk_residual_stream_incorrect_hint_2_q0-1139_tokens1_20251216_224948.json \\
        --split test
        """
    )
    
    parser.add_argument('probe_json', type=str,
                        help='Path to trained probe JSON file')
    parser.add_argument('categorized_json', type=str,
                        help='Path to categorized data JSON for evaluation')
    parser.add_argument('bulk_json', type=str,
                        help='Path to bulk residual stream JSON for evaluation')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default=None,
                        help='Evaluate only this split (default: evaluate all)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for results JSON')
    
    args = parser.parse_args()
    
    try:
        run_cross_mode_evaluation(
            probe_path=args.probe_json,
            categorized_path=args.categorized_json,
            bulk_path=args.bulk_json,
            split=args.split,
            output_path=args.output
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

