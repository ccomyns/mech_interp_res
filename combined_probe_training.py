#!/usr/bin/env python3
"""
Combined Linear Probe Training

This script trains a single linear probe on combined data from two different
incorrect hint modes, then evaluates on the combined test data.

The goal is to learn a more robust, format-invariant representation of
"correct answer vs agrees with hint" that generalizes across hint formats.

Usage:
    python combined_probe_training.py <categorized_json_1> <categorized_json_2> [options]

Example:
    python combined_probe_training.py \
        linear_probe_mode_1/categorized_data_no_math_20251215_234715.json \
        linear_probe_mode_2/categorized_data_no_math_20251217_150300.json \
        --layer 18
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProbeResults:
    """Results from training a linear probe."""
    accuracy: float
    balanced_accuracy: float
    cross_val_accuracy: float
    cross_val_std: float
    classification_report: str
    confusion_matrix: np.ndarray
    probe_weights: np.ndarray
    probe_bias: float
    feature_dim: int
    num_train_samples: int
    num_test_samples: int


def resolve_bulk_files_from_categorized(categorized_path: str) -> List[str]:
    """
    Trace from categorized JSON → cleaned answers JSON(s) → bulk residual stream JSON(s).
    
    Args:
        categorized_path: Path to categorized JSON file
        
    Returns:
        List of paths to bulk residual stream JSON files
    """
    categorized_dir = Path(categorized_path).parent
    
    with open(categorized_path, 'r') as f:
        categorized_data = json.load(f)
    
    source_files = categorized_data.get('metadata', {}).get('source_files', [])
    
    bulk_files = []
    
    for source_file in source_files:
        # Resolve relative paths
        if not Path(source_file).is_absolute():
            # Try relative to categorized file's directory
            cleaned_path = categorized_dir / Path(source_file).name
            if not cleaned_path.exists():
                # Try relative to workspace root
                cleaned_path = Path(source_file)
                if not cleaned_path.exists():
                    # Try looking in parent directory
                    cleaned_path = categorized_dir.parent / source_file
        else:
            cleaned_path = Path(source_file)
        
        if not cleaned_path.exists():
            print(f"⚠️  Warning: Cleaned answers file not found: {source_file}")
            print(f"   Tried: {cleaned_path}")
            continue
        
        # Load cleaned answers to get bulk file path
        with open(cleaned_path, 'r') as f:
            cleaned_data = json.load(f)
        
        bulk_source = cleaned_data.get('metadata', {}).get('source_file', '')
        
        if bulk_source:
            # Resolve bulk file path
            if not Path(bulk_source).is_absolute():
                # Try relative to cleaned answers file's directory
                bulk_path = cleaned_path.parent / Path(bulk_source).name
                if not bulk_path.exists():
                    # Try relative to workspace root
                    bulk_path = Path(bulk_source)
                    if not bulk_path.exists():
                        # Try looking in parent directory
                        bulk_path = cleaned_path.parent.parent / bulk_source
            else:
                bulk_path = Path(bulk_source)
            
            if bulk_path.exists():
                bulk_files.append(str(bulk_path))
            else:
                print(f"⚠️  Warning: Bulk file not found: {bulk_source}")
                print(f"   Tried: {bulk_path}")
    
    return bulk_files


def load_categorized_data(json_path: str) -> Tuple[Dict[str, Any], Dict[str, List[Dict]]]:
    """Load categorized data from JSON file."""
    print(f"\n📂 Loading categorized data from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    statistics = data.get('statistics', {})
    
    categorized_data = {
        'correct_answer': data.get('correct_answer', []),
        'agrees_with_hint': data.get('agrees_with_hint', [])
    }
    
    print(f"✅ Loaded categorized data")
    print(f"   correct_answer: {len(categorized_data['correct_answer'])}")
    print(f"   agrees_with_hint: {len(categorized_data['agrees_with_hint'])}")
    
    return metadata, categorized_data


def load_residual_streams(bulk_files: List[str], 
                          question_indices: List[int]) -> Dict[int, Dict[str, Any]]:
    """Load residual stream data for specific question indices from bulk files."""
    print(f"\n📂 Loading residual streams from {len(bulk_files)} bulk file(s)...")
    
    target_indices = set(question_indices)
    residual_streams = {}
    
    for bulk_file in bulk_files:
        print(f"   Loading: {Path(bulk_file).name}")
        with open(bulk_file, 'r', encoding='utf-8') as f:
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
        print(f"⚠️  Warning: {len(missing)} questions not found in bulk files")
    
    return residual_streams


def extract_features(categorized_data: Dict[str, List[Dict]],
                     residual_streams: Dict[int, Dict[str, Any]],
                     layer: int,
                     split: str,
                     mode_label: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Extract feature vectors and labels from categorized data.
    
    Returns:
        Tuple of (features, labels, question_indices, mode_labels)
    """
    features = []
    labels = []
    question_indices = []
    mode_labels = []
    
    layer_key = f"layer_{layer}"
    
    # Process correct_answer category (label = -1)
    for item in categorized_data['correct_answer']:
        if item.get('split') != split:
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
        labels.append(-1)
        question_indices.append(q_idx)
        mode_labels.append(mode_label)
    
    # Process agrees_with_hint category (label = +1)
    for item in categorized_data['agrees_with_hint']:
        if item.get('split') != split:
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
        mode_labels.append(mode_label)
    
    return np.array(features), np.array(labels), question_indices, mode_labels


def train_linear_probe(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       random_state: int = 42,
                       n_cv_folds: int = 5,
                       regularization: float = 1.0) -> Tuple[ProbeResults, StandardScaler, LogisticRegression]:
    """Train a linear probe and return results along with the trained model."""
    np.random.seed(random_state)
    
    n_samples = len(y_train)
    if n_samples < 10:
        raise ValueError(f"Not enough samples for training: {n_samples}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    probe = LogisticRegression(
        C=regularization,
        max_iter=1000,
        solver='saga',
        random_state=random_state
    )
    probe.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = probe.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=min(n_cv_folds, n_samples // 2), shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=cv)
    
    report = classification_report(y_test, y_pred, 
                                   target_names=['Correct Answer (-1)', 'Agrees with Hint (+1)'])
    cm = confusion_matrix(y_test, y_pred)
    
    results = ProbeResults(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        cross_val_accuracy=cv_scores.mean(),
        cross_val_std=cv_scores.std(),
        classification_report=report,
        confusion_matrix=cm,
        probe_weights=probe.coef_[0],
        probe_bias=probe.intercept_[0],
        feature_dim=X_train.shape[1],
        num_train_samples=len(y_train),
        num_test_samples=len(y_test)
    )
    
    return results, scaler, probe


def print_results(results: ProbeResults, 
                  train_mode_counts: Dict[str, Dict[str, int]],
                  test_mode_counts: Dict[str, Dict[str, int]]):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"COMBINED LINEAR PROBE TRAINING RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📊 Training Data Distribution:")
    for mode, counts in train_mode_counts.items():
        print(f"   {mode}:")
        print(f"      Correct Answer (-1): {counts.get(-1, 0)}")
        print(f"      Agrees with Hint (+1): {counts.get(1, 0)}")
    
    print(f"\n📊 Test Data Distribution:")
    for mode, counts in test_mode_counts.items():
        print(f"   {mode}:")
        print(f"      Correct Answer (-1): {counts.get(-1, 0)}")
        print(f"      Agrees with Hint (+1): {counts.get(1, 0)}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Test Accuracy: {results.accuracy:.4f} ({results.accuracy*100:.2f}%)")
    print(f"   Balanced Accuracy: {results.balanced_accuracy:.4f} ({results.balanced_accuracy*100:.2f}%)")
    print(f"   Cross-Val Accuracy: {results.cross_val_accuracy:.4f} ± {results.cross_val_std:.4f}")
    
    print(f"\n📋 Dataset Info:")
    print(f"   Feature dimension: {results.feature_dim}")
    print(f"   Training samples: {results.num_train_samples}")
    print(f"   Test samples: {results.num_test_samples}")
    
    print(f"\n📊 Classification Report:")
    print(results.classification_report)
    
    print(f"📊 Confusion Matrix:")
    print(f"   Predicted:  -1    +1")
    print(f"   Actual -1: {results.confusion_matrix[0][0]:4d}  {results.confusion_matrix[0][1]:4d}")
    print(f"   Actual +1: {results.confusion_matrix[1][0]:4d}  {results.confusion_matrix[1][1]:4d}")
    
    print(f"{'='*80}")


def evaluate_per_mode(probe: LogisticRegression, 
                      scaler: StandardScaler,
                      X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      test_modes: List[str]):
    """Evaluate the probe separately for each mode."""
    print(f"\n{'='*80}")
    print(f"PER-MODE EVALUATION")
    print(f"{'='*80}")
    
    unique_modes = list(set(test_modes))
    X_test_scaled = scaler.transform(X_test)
    
    for mode in sorted(unique_modes):
        mask = np.array([m == mode for m in test_modes])
        X_mode = X_test_scaled[mask]
        y_mode = y_test[mask]
        
        if len(y_mode) == 0:
            continue
        
        y_pred = probe.predict(X_mode)
        accuracy = accuracy_score(y_mode, y_pred)
        balanced_acc = balanced_accuracy_score(y_mode, y_pred)
        
        print(f"\n   {mode}:")
        print(f"      Samples: {len(y_mode)} (correct: {sum(y_mode == -1)}, hint: {sum(y_mode == 1)})")
        print(f"      Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"      Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    
    print(f"{'='*80}")


def save_results(results: ProbeResults, 
                 output_path: str,
                 config: Dict,
                 train_mode_counts: Dict,
                 test_mode_counts: Dict):
    """Save results to JSON."""
    output_data = {
        'metadata': {
            'training_date': datetime.now().isoformat(),
            'mode1_categorized': config['mode1_path'],
            'mode2_categorized': config['mode2_path'],
            'layer': config['layer'],
            'random_seed': config['seed']
        },
        'train_distribution': train_mode_counts,
        'test_distribution': test_mode_counts,
        'results': {
            'accuracy': results.accuracy,
            'balanced_accuracy': results.balanced_accuracy,
            'cross_val_accuracy': results.cross_val_accuracy,
            'cross_val_std': results.cross_val_std,
            'feature_dim': results.feature_dim,
            'num_train_samples': results.num_train_samples,
            'num_test_samples': results.num_test_samples,
            'confusion_matrix': results.confusion_matrix.tolist()
        },
        'probe_weights': results.probe_weights.tolist(),
        'probe_bias': results.probe_bias
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved results to: {output_path}")


def run_combined_training(mode1_path: str,
                          mode2_path: str,
                          layer: int = 18,
                          random_state: int = 42,
                          output_path: Optional[str] = None):
    """
    Run combined linear probe training on data from two modes.
    """
    print("="*80)
    print("COMBINED LINEAR PROBE TRAINING")
    print("="*80)
    print(f"\nMode 1: {mode1_path}")
    print(f"Mode 2: {mode2_path}")
    print(f"Layer: {layer}")
    
    # Load categorized data for both modes
    _, cat_data_1 = load_categorized_data(mode1_path)
    _, cat_data_2 = load_categorized_data(mode2_path)
    
    # Resolve bulk files for each mode
    print("\n🔍 Resolving bulk residual stream files...")
    bulk_files_1 = resolve_bulk_files_from_categorized(mode1_path)
    bulk_files_2 = resolve_bulk_files_from_categorized(mode2_path)
    
    print(f"   Mode 1 bulk files: {[Path(f).name for f in bulk_files_1]}")
    print(f"   Mode 2 bulk files: {[Path(f).name for f in bulk_files_2]}")
    
    # Get all question indices for each mode
    mode1_indices = []
    for item in cat_data_1['correct_answer'] + cat_data_1['agrees_with_hint']:
        mode1_indices.append(item['question_idx'])
    
    mode2_indices = []
    for item in cat_data_2['correct_answer'] + cat_data_2['agrees_with_hint']:
        mode2_indices.append(item['question_idx'])
    
    # Load residual streams
    residual_streams_1 = load_residual_streams(bulk_files_1, mode1_indices)
    residual_streams_2 = load_residual_streams(bulk_files_2, mode2_indices)
    
    # Extract features for training
    print(f"\n🔄 Extracting training features from layer {layer}...")
    
    X_train_1, y_train_1, _, modes_train_1 = extract_features(
        cat_data_1, residual_streams_1, layer, 'train', 'mode1'
    )
    X_train_2, y_train_2, _, modes_train_2 = extract_features(
        cat_data_2, residual_streams_2, layer, 'train', 'mode2'
    )
    
    # Combine training data
    X_train = np.vstack([X_train_1, X_train_2]) if len(X_train_1) > 0 and len(X_train_2) > 0 else (X_train_1 if len(X_train_1) > 0 else X_train_2)
    y_train = np.concatenate([y_train_1, y_train_2])
    train_modes = modes_train_1 + modes_train_2
    
    print(f"✅ Combined training data: {len(X_train)} samples")
    print(f"   Mode 1: {len(X_train_1)} samples (correct: {sum(y_train_1 == -1)}, hint: {sum(y_train_1 == 1)})")
    print(f"   Mode 2: {len(X_train_2)} samples (correct: {sum(y_train_2 == -1)}, hint: {sum(y_train_2 == 1)})")
    
    # Extract features for testing
    print(f"\n🔄 Extracting test features from layer {layer}...")
    
    X_test_1, y_test_1, _, modes_test_1 = extract_features(
        cat_data_1, residual_streams_1, layer, 'test', 'mode1'
    )
    X_test_2, y_test_2, _, modes_test_2 = extract_features(
        cat_data_2, residual_streams_2, layer, 'test', 'mode2'
    )
    
    # Combine test data
    X_test = np.vstack([X_test_1, X_test_2]) if len(X_test_1) > 0 and len(X_test_2) > 0 else (X_test_1 if len(X_test_1) > 0 else X_test_2)
    y_test = np.concatenate([y_test_1, y_test_2])
    test_modes = modes_test_1 + modes_test_2
    
    print(f"✅ Combined test data: {len(X_test)} samples")
    print(f"   Mode 1: {len(X_test_1)} samples (correct: {sum(y_test_1 == -1)}, hint: {sum(y_test_1 == 1)})")
    print(f"   Mode 2: {len(X_test_2)} samples (correct: {sum(y_test_2 == -1)}, hint: {sum(y_test_2 == 1)})")
    
    # Train probe
    print("\n" + "="*80)
    print("TRAINING COMBINED LINEAR PROBE")
    print("="*80)
    
    results, scaler, probe = train_linear_probe(X_train, y_train, X_test, y_test, random_state)
    
    # Compute mode counts for reporting
    train_mode_counts = {
        'mode1': {-1: int(sum(y_train_1 == -1)), 1: int(sum(y_train_1 == 1))},
        'mode2': {-1: int(sum(y_train_2 == -1)), 1: int(sum(y_train_2 == 1))}
    }
    test_mode_counts = {
        'mode1': {-1: int(sum(y_test_1 == -1)), 1: int(sum(y_test_1 == 1))},
        'mode2': {-1: int(sum(y_test_2 == -1)), 1: int(sum(y_test_2 == 1))}
    }
    
    # Print results
    print_results(results, train_mode_counts, test_mode_counts)
    
    # Per-mode evaluation
    evaluate_per_mode(probe, scaler, X_test, y_test, test_modes)
    
    # Save results
    if output_path:
        config = {
            'mode1_path': mode1_path,
            'mode2_path': mode2_path,
            'layer': layer,
            'seed': random_state
        }
        save_results(results, output_path, config, train_mode_counts, test_mode_counts)
    
    return results, scaler, probe


def main():
    parser = argparse.ArgumentParser(
        description="Train a combined linear probe on data from two incorrect hint modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python combined_probe_training.py \\
        linear_probe_mode_1/categorized_data_no_math_20251215_234715.json \\
        linear_probe_mode_2/categorized_data_no_math_20251217_150300.json

    # Specify layer and output
    python combined_probe_training.py \\
        linear_probe_mode_1/categorized_data_no_math_20251215_234715.json \\
        linear_probe_mode_2/categorized_data_no_math_20251217_150300.json \\
        --layer 18 \\
        -o combined_probe_results.json
        """
    )
    
    parser.add_argument('mode1_categorized', type=str,
                        help='Path to categorized JSON for mode 1')
    parser.add_argument('mode2_categorized', type=str,
                        help='Path to categorized JSON for mode 2')
    parser.add_argument('--layer', type=int, default=18,
                        help='Layer to extract features from (default: 18)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for results JSON')
    
    args = parser.parse_args()
    
    try:
        run_combined_training(
            mode1_path=args.mode1_categorized,
            mode2_path=args.mode2_categorized,
            layer=args.layer,
            random_state=args.seed,
            output_path=args.output
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

