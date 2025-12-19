#!/usr/bin/env python3
"""
Zebra Puzzle Linear Probe Training

This script trains a linear probe on zebra puzzle data to distinguish between:
- Correct solutions (partial_score = 1.0, label = -1)
- Fooled by red herrings (partial_score = 0.333, label = +1)

Usage:
    python zebra_linear_probe_training.py <zebra_residual_json> [options]

Example:
    python zebra_linear_probe_training.py \
        zebra_puzzles/zebra_residual_test_n1000_tokens1_20251218_225833.json \
        --layer 18 \
        --train-ratio 0.7 \
        -o probe_results/zebra_probe_layer18.json
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
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


def load_zebra_data(zebra_path: str, layer: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load zebra puzzle data and extract features for specific partial scores.
    
    Maps:
    - partial_score == 1.0 -> label -1 (correct_answer)
    - partial_score == 0.333... -> label +1 (fooled by red herrings)
    
    Args:
        zebra_path: Path to zebra residual stream JSON
        layer: Layer to extract features from
        
    Returns:
        Tuple of (features array, labels array, puzzle indices)
    """
    print(f"\n📂 Loading zebra puzzle data from: {zebra_path}")
    
    with open(zebra_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    print(f"   Total puzzles in file: {len(results)}")
    
    features = []
    labels = []
    puzzle_indices = []
    
    layer_key = f"layer_{layer}"
    
    # Target partial scores
    CORRECT_SCORE = 1.0
    HINT_SCORE = 1/3  # 0.333...
    TOLERANCE = 0.01  # Allow small floating point differences
    
    correct_count = 0
    hint_count = 0
    skipped_count = 0
    
    for result in results:
        partial_score = result.get('partial_score', -1)
        puzzle_idx = result.get('puzzle_idx')
        
        # Check if this is a score we want
        is_correct = abs(partial_score - CORRECT_SCORE) < TOLERANCE
        is_hint = abs(partial_score - HINT_SCORE) < TOLERANCE
        
        if not is_correct and not is_hint:
            skipped_count += 1
            continue
        
        # Get hidden states
        hidden_states = result.get('residual_stream_hidden_states', {})
        if layer_key not in hidden_states:
            continue
        
        layer_data = hidden_states[layer_key]
        if not layer_data:
            continue
        
        # Get the first position available
        position_key = list(layer_data.keys())[0]
        hidden_vector = np.array(layer_data[position_key])
        
        features.append(hidden_vector)
        puzzle_indices.append(puzzle_idx)
        
        if is_correct:
            labels.append(-1)  # correct_answer
            correct_count += 1
        else:
            labels.append(+1)  # fooled by red herrings
            hint_count += 1
    
    print(f"✅ Loaded zebra puzzle data")
    print(f"   Correct (partial_score=1.0, label=-1): {correct_count}")
    print(f"   Fooled (partial_score~0.33, label=+1): {hint_count}")
    print(f"   Skipped (other partial scores): {skipped_count}")
    print(f"   Total usable samples: {correct_count + hint_count}")
    
    return np.array(features), np.array(labels), puzzle_indices


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
    
    print(f"\n🔧 Training configuration:")
    print(f"   Training samples: {n_samples}")
    print(f"   Test samples: {len(y_test)}")
    print(f"   Feature dimension: {X_train.shape[1]}")
    print(f"   Regularization (C): {regularization}")
    print(f"   CV folds: {n_cv_folds}")
    
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
                                   target_names=['Correct (-1)', 'Fooled (+1)'])
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


def print_results(results: ProbeResults):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"ZEBRA PUZZLE LINEAR PROBE TRAINING RESULTS")
    print(f"{'='*80}")
    
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


def save_results(results: ProbeResults, 
                 output_path: str,
                 config: Dict):
    """Save results to JSON."""
    output_data = {
        'metadata': {
            'training_date': datetime.now().isoformat(),
            'zebra_file': config['zebra_path'],
            'layer': config['layer'],
            'train_ratio': config['train_ratio'],
            'random_seed': config['seed'],
            'regularization': config['regularization'],
            'balance_classes': config.get('balance_classes', True)
        },
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
        'probe_bias': float(results.probe_bias),
        'layer': config['layer']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved results to: {output_path}")


def run_zebra_training(zebra_path: str,
                       layer: int = 18,
                       train_ratio: float = 0.7,
                       random_state: int = 42,
                       regularization: float = 1.0,
                       n_cv_folds: int = 5,
                       balance_classes: bool = True,
                       output_path: Optional[str] = None):
    """
    Run linear probe training on zebra puzzle data.
    
    Args:
        zebra_path: Path to zebra residual stream JSON
        layer: Layer to extract features from
        train_ratio: Ratio of data to use for training (rest for testing)
        random_state: Random seed for reproducibility
        regularization: Regularization parameter (C) for logistic regression
        n_cv_folds: Number of cross-validation folds
        balance_classes: Whether to balance classes by undersampling majority class
        output_path: Optional path to save results
    """
    print("="*80)
    print("ZEBRA PUZZLE LINEAR PROBE TRAINING")
    print("="*80)
    print(f"\nZebra data: {zebra_path}")
    print(f"Layer: {layer}")
    print(f"Train ratio: {train_ratio}")
    print(f"Balance classes: {balance_classes}")
    print(f"Random seed: {random_state}")
    
    # Load zebra data
    X, y, puzzle_indices = load_zebra_data(zebra_path, layer)
    
    if len(X) == 0:
        raise ValueError("No valid features extracted from zebra data!")
    
    # Check class balance
    n_correct = sum(y == -1)
    n_fooled = sum(y == 1)
    print(f"\n📊 Original class distribution:")
    print(f"   Correct (-1): {n_correct} ({100*n_correct/len(y):.1f}%)")
    print(f"   Fooled (+1): {n_fooled} ({100*n_fooled/len(y):.1f}%)")
    
    # Balance classes if requested
    if balance_classes:
        print(f"\n⚖️  Balancing classes by undersampling majority class...")
        np.random.seed(random_state)
        
        # Find indices for each class
        correct_indices = np.where(y == -1)[0]
        fooled_indices = np.where(y == 1)[0]
        
        # Determine the size of the smaller class
        min_class_size = min(len(correct_indices), len(fooled_indices))
        
        # Randomly sample from each class
        correct_sampled = np.random.choice(correct_indices, size=min_class_size, replace=False)
        fooled_sampled = np.random.choice(fooled_indices, size=min_class_size, replace=False)
        
        # Combine and shuffle
        balanced_indices = np.concatenate([correct_sampled, fooled_sampled])
        np.random.shuffle(balanced_indices)
        
        # Apply balancing
        X = X[balanced_indices]
        y = y[balanced_indices]
        puzzle_indices = [puzzle_indices[i] for i in balanced_indices]
        
        print(f"   Balanced to {min_class_size} samples per class")
        print(f"   Total samples after balancing: {len(X)}")
        print(f"   Correct (-1): {sum(y == -1)} ({100*sum(y == -1)/len(y):.1f}%)")
        print(f"   Fooled (+1): {sum(y == 1)} ({100*sum(y == 1)/len(y):.1f}%)")
    
    # Split into train and test
    print(f"\n✂️  Splitting data (train ratio: {train_ratio})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=random_state, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples (correct: {sum(y_train == -1)}, fooled: {sum(y_train == 1)})")
    print(f"   Test: {len(X_test)} samples (correct: {sum(y_test == -1)}, fooled: {sum(y_test == 1)})")
    
    # Train probe
    print(f"\n{'='*80}")
    print("TRAINING LINEAR PROBE")
    print(f"{'='*80}")
    
    results, scaler, probe = train_linear_probe(
        X_train, y_train, X_test, y_test, 
        random_state, n_cv_folds, regularization
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if output_path:
        config = {
            'zebra_path': zebra_path,
            'layer': layer,
            'train_ratio': train_ratio,
            'seed': random_state,
            'regularization': regularization,
            'balance_classes': balance_classes
        }
        save_results(results, output_path, config)
    
    return results, scaler, probe


def main():
    parser = argparse.ArgumentParser(
        description="Train a linear probe on zebra puzzle data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (with balanced classes by default)
    python zebra_linear_probe_training.py \\
        zebra_puzzles/zebra_residual_test_n1000_tokens1_20251218_225833.json

    # Specify layer and output
    python zebra_linear_probe_training.py \\
        zebra_puzzles/zebra_residual_test_n1000_tokens1_20251218_225833.json \\
        --layer 18 \\
        --train-ratio 0.7 \\
        -o probe_results/zebra_probe_layer18.json

    # Train without balancing classes (use original imbalanced data)
    python zebra_linear_probe_training.py \\
        zebra_puzzles/zebra_residual_test_n1000_tokens1_20251218_225833.json \\
        --no-balance \\
        -o probe_results/zebra_probe_layer18_imbalanced.json

    # Adjust regularization
    python zebra_linear_probe_training.py \\
        zebra_puzzles/zebra_residual_test_n1000_tokens1_20251218_225833.json \\
        --regularization 0.1 \\
        -o probe_results/zebra_probe_layer18_c0.1.json
        """
    )
    
    parser.add_argument('zebra_json', type=str,
                        help='Path to zebra residual stream JSON file')
    parser.add_argument('--layer', type=int, default=18,
                        help='Layer to extract features from (default: 18)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--regularization', type=float, default=1.0,
                        help='Regularization parameter C (default: 1.0)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--no-balance', action='store_false', dest='balance_classes',
                        help='Do not balance classes (default: classes are balanced)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for results JSON')
    
    args = parser.parse_args()
    
    # Validate train_ratio
    if not 0 < args.train_ratio < 1:
        parser.error("train-ratio must be between 0 and 1")
    
    try:
        run_zebra_training(
            zebra_path=args.zebra_json,
            layer=args.layer,
            train_ratio=args.train_ratio,
            random_state=args.seed,
            regularization=args.regularization,
            n_cv_folds=args.cv_folds,
            balance_classes=args.balance_classes,
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

