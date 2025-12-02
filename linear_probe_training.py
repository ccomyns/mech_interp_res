#!/usr/bin/env python3
"""
Linear Probe Training for Residual Stream Analysis

This script trains two linear probes on residual stream hidden states extracted
from Gemma-3-4B's responses to MMLU questions with incorrect hints.

Data Categories:
    1. correct_answer: Model chose the correct answer
    2. incorrect_not_hint: Model chose an incorrect answer that is NOT the hint
    3. chose_hint: Model chose the incorrect hint answer
    4. exceeded_budget: Model hit the token limit

Probe 1: Agrees vs Against Hint
    - agrees_with_hint (+1): chose_hint
    - went_against_hint (-1): correct_answer, incorrect_not_hint
    - Excludes: exceeded_budget

Probe 2: Distracted vs Found Answer
    - distracted_by_hint (+1): exceeded_budget
    - found_answer (-1): correct_answer, incorrect_not_hint, chose_hint

Usage:
    python linear_probe_training.py <labeled_json_path> [options]

Example:
    python linear_probe_training.py hidden_values/labeled_results.json --layer 1 --position 114
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LabeledExample:
    """A single labeled example for probe training."""
    question_idx: int
    category: str  # 'correct_answer', 'incorrect_not_hint', 'chose_hint', 'exceeded_budget'
    hidden_states: Dict[str, np.ndarray]  # layer_key -> position -> hidden state
    metadata: Dict[str, Any]  # Additional info about the example


@dataclass
class ProbeResults:
    """Results from training a linear probe."""
    accuracy: float
    cross_val_accuracy: float
    cross_val_std: float
    classification_report: str
    confusion_matrix: np.ndarray
    probe_weights: np.ndarray
    probe_bias: float
    feature_dim: int
    num_train_samples: int
    num_test_samples: int


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_labeled_data(json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load labeled data from JSON file.
    
    Expected JSON structure:
    {
        "metadata": {...},
        "results": [
            {
                "question_idx": 0,
                "category": "correct_answer" | "incorrect_not_hint" | "chose_hint" | "exceeded_budget",
                "residual_stream_hidden_states": {
                    "layer_1": {"114": [...], ...},
                    "layer_18": {"114": [...], ...}
                },
                ...
            },
            ...
        ]
    }
    """
    print(f"\n📂 Loading labeled data from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    print(f"✅ Loaded {len(results)} examples")
    
    # Count categories
    category_counts = {}
    for result in results:
        cat = result.get('category', 'unknown')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n📊 Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat}: {count}")
    
    return metadata, results


def extract_features(results: List[Dict[str, Any]], 
                     layer: int, 
                     position: int) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Extract feature vectors from residual stream hidden states.
    
    Args:
        results: List of result dictionaries from JSON
        layer: Which layer to extract (e.g., 1 or 18)
        position: Which token position to extract
        
    Returns:
        Tuple of (features array, categories list, question indices list)
    """
    features = []
    categories = []
    question_indices = []
    
    layer_key = f"layer_{layer}"
    position_key = str(position)
    
    for result in results:
        if 'error' in result:
            continue
            
        hidden_states = result.get('residual_stream_hidden_states', {})
        
        if layer_key not in hidden_states:
            print(f"⚠️  Layer {layer} not found for question {result.get('question_idx', '?')}")
            continue
            
        if position_key not in hidden_states[layer_key]:
            print(f"⚠️  Position {position} not found for question {result.get('question_idx', '?')}")
            continue
        
        hidden_vector = np.array(hidden_states[layer_key][position_key])
        category = result.get('category', 'unknown')
        
        features.append(hidden_vector)
        categories.append(category)
        question_indices.append(result.get('question_idx', -1))
    
    return np.array(features), categories, question_indices


def get_available_layers_and_positions(results: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """Get the available layers and positions from the data."""
    layers = set()
    positions = set()
    
    for result in results:
        if 'error' in result:
            continue
        hidden_states = result.get('residual_stream_hidden_states', {})
        for layer_key, pos_dict in hidden_states.items():
            if layer_key.startswith('layer_'):
                layers.add(int(layer_key.split('_')[1]))
                for pos_key in pos_dict.keys():
                    positions.add(int(pos_key))
    
    return sorted(layers), sorted(positions)


# ============================================================================
# Probe Training Functions
# ============================================================================

def prepare_probe1_data(features: np.ndarray, 
                        categories: List[str],
                        question_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Prepare data for Probe 1: Agrees vs Against Hint
    
    Labels:
        +1 (agrees_with_hint): chose_hint
        -1 (went_against_hint): correct_answer, incorrect_not_hint
        
    Excludes: exceeded_budget
    """
    X = []
    y = []
    indices = []
    
    for i, (feat, cat, idx) in enumerate(zip(features, categories, question_indices)):
        if cat == 'exceeded_budget':
            continue  # Exclude from this probe
        elif cat == 'chose_hint':
            X.append(feat)
            y.append(1)  # agrees_with_hint
            indices.append(idx)
        elif cat in ['correct_answer', 'incorrect_not_hint']:
            X.append(feat)
            y.append(-1)  # went_against_hint
            indices.append(idx)
    
    return np.array(X), np.array(y), indices


def prepare_probe2_data(features: np.ndarray,
                        categories: List[str],
                        question_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Prepare data for Probe 2: Distracted vs Found Answer
    
    Labels:
        +1 (distracted_by_hint): exceeded_budget
        -1 (found_answer): correct_answer, incorrect_not_hint, chose_hint
    """
    X = []
    y = []
    indices = []
    
    for i, (feat, cat, idx) in enumerate(zip(features, categories, question_indices)):
        if cat == 'exceeded_budget':
            X.append(feat)
            y.append(1)  # distracted_by_hint
            indices.append(idx)
        elif cat in ['correct_answer', 'incorrect_not_hint', 'chose_hint']:
            X.append(feat)
            y.append(-1)  # found_answer
            indices.append(idx)
    
    return np.array(X), np.array(y), indices


def train_linear_probe(X: np.ndarray, 
                       y: np.ndarray,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       n_cv_folds: int = 5,
                       regularization: float = 1.0) -> ProbeResults:
    """
    Train a linear probe (logistic regression) on the data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (+1 or -1)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        n_cv_folds: Number of cross-validation folds
        regularization: L2 regularization strength (C parameter)
        
    Returns:
        ProbeResults object with training results
    """
    # Check if we have enough samples
    n_samples = len(y)
    if n_samples < 10:
        raise ValueError(f"Not enough samples for training: {n_samples}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train logistic regression
    probe = LogisticRegression(
        C=regularization,
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state
    )
    probe.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(probe, X_scaled, y, cv=min(n_cv_folds, n_samples // 2))
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Negative (-1)', 'Positive (+1)'])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return ProbeResults(
        accuracy=accuracy,
        cross_val_accuracy=cv_scores.mean(),
        cross_val_std=cv_scores.std(),
        classification_report=report,
        confusion_matrix=cm,
        probe_weights=probe.coef_[0],
        probe_bias=probe.intercept_[0],
        feature_dim=X.shape[1],
        num_train_samples=len(y_train),
        num_test_samples=len(y_test)
    )


# ============================================================================
# Results Display
# ============================================================================

def print_probe_results(results: ProbeResults, probe_name: str, label_descriptions: Dict[int, str]):
    """Print formatted results for a probe."""
    print(f"\n{'='*80}")
    print(f"PROBE RESULTS: {probe_name}")
    print(f"{'='*80}")
    
    print(f"\n📊 Label Mapping:")
    for label, desc in label_descriptions.items():
        print(f"   {label:+d}: {desc}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Test Accuracy: {results.accuracy:.4f} ({results.accuracy*100:.2f}%)")
    print(f"   Cross-Val Accuracy: {results.cross_val_accuracy:.4f} ± {results.cross_val_std:.4f}")
    
    print(f"\n📋 Dataset Info:")
    print(f"   Feature dimension: {results.feature_dim}")
    print(f"   Training samples: {results.num_train_samples}")
    print(f"   Test samples: {results.num_test_samples}")
    
    print(f"\n📊 Classification Report:")
    print(results.classification_report)
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   Predicted:  -1    +1")
    print(f"   Actual -1: {results.confusion_matrix[0][0]:4d}  {results.confusion_matrix[0][1]:4d}")
    print(f"   Actual +1: {results.confusion_matrix[1][0]:4d}  {results.confusion_matrix[1][1]:4d}")
    
    # Top features by weight magnitude
    top_k = 10
    weight_magnitudes = np.abs(results.probe_weights)
    top_indices = np.argsort(weight_magnitudes)[-top_k:][::-1]
    
    print(f"\n🔍 Top {top_k} Features by Weight Magnitude:")
    print(f"   {'Index':<8} {'Weight':<12} {'|Weight|'}")
    print(f"   {'-'*32}")
    for idx in top_indices:
        print(f"   {idx:<8} {results.probe_weights[idx]:+.6f}  {weight_magnitudes[idx]:.6f}")
    
    print(f"\n   Probe bias: {results.probe_bias:+.6f}")
    print(f"{'='*80}")


def save_probe_results(results: ProbeResults, 
                       probe_name: str,
                       output_path: str,
                       layer: int,
                       position: int,
                       label_descriptions: Dict[int, str]):
    """Save probe results to JSON file."""
    output_data = {
        'probe_name': probe_name,
        'layer': layer,
        'position': position,
        'label_descriptions': {str(k): v for k, v in label_descriptions.items()},
        'accuracy': results.accuracy,
        'cross_val_accuracy': results.cross_val_accuracy,
        'cross_val_std': results.cross_val_std,
        'feature_dim': results.feature_dim,
        'num_train_samples': results.num_train_samples,
        'num_test_samples': results.num_test_samples,
        'confusion_matrix': results.confusion_matrix.tolist(),
        'probe_weights': results.probe_weights.tolist(),
        'probe_bias': results.probe_bias
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Saved probe results to: {output_path}")


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiments(json_path: str,
                    layer: Optional[int] = None,
                    position: Optional[int] = None,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    save_results: bool = True,
                    output_dir: str = "probe_results"):
    """
    Run both linear probe experiments.
    
    Args:
        json_path: Path to labeled JSON data
        layer: Layer to extract features from (auto-detected if None)
        position: Position to extract features from (auto-detected if None)
        test_size: Fraction of data for testing
        random_state: Random seed
        save_results: Whether to save results to files
        output_dir: Directory for output files
    """
    print("="*80)
    print("LINEAR PROBE TRAINING EXPERIMENTS")
    print("="*80)
    
    # Load data
    metadata, results = load_labeled_data(json_path)
    
    # Auto-detect layer and position if not specified
    available_layers, available_positions = get_available_layers_and_positions(results)
    print(f"\n📍 Available layers: {available_layers}")
    print(f"📍 Available positions: {available_positions}")
    
    if layer is None:
        layer = available_layers[0] if available_layers else 1
        print(f"   Using layer: {layer} (auto-selected)")
    
    if position is None:
        position = available_positions[0] if available_positions else 114
        print(f"   Using position: {position} (auto-selected)")
    
    # Extract features
    print(f"\n🔄 Extracting features from layer {layer}, position {position}...")
    features, categories, question_indices = extract_features(results, layer, position)
    
    if len(features) == 0:
        raise ValueError("No valid features extracted. Check layer and position settings.")
    
    print(f"✅ Extracted {len(features)} feature vectors of dimension {features.shape[1]}")
    
    # ========================================================================
    # PROBE 1: Agrees vs Against Hint
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING PROBE 1: Agrees vs Against Hint")
    print("="*80)
    
    X1, y1, idx1 = prepare_probe1_data(features, categories, question_indices)
    
    print(f"\n📊 Probe 1 Data:")
    print(f"   Total samples: {len(y1)}")
    print(f"   Agrees with hint (+1): {sum(y1 == 1)}")
    print(f"   Went against hint (-1): {sum(y1 == -1)}")
    
    if len(y1) < 10:
        print("⚠️  Not enough samples for Probe 1. Skipping...")
        probe1_results = None
    elif len(set(y1)) < 2:
        print("⚠️  Only one class present in Probe 1 data. Skipping...")
        probe1_results = None
    else:
        probe1_results = train_linear_probe(X1, y1, test_size, random_state)
        
        probe1_labels = {
            1: "agrees_with_hint (chose the incorrect hint)",
            -1: "went_against_hint (chose correct or other incorrect)"
        }
        print_probe_results(probe1_results, "Agrees vs Against Hint", probe1_labels)
        
        if save_results:
            import os
            os.makedirs(output_dir, exist_ok=True)
            save_probe_results(
                probe1_results,
                "agrees_vs_against_hint",
                f"{output_dir}/probe1_agrees_vs_against_layer{layer}_pos{position}.json",
                layer, position, probe1_labels
            )
    
    # ========================================================================
    # PROBE 2: Distracted vs Found Answer
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING PROBE 2: Distracted vs Found Answer")
    print("="*80)
    
    X2, y2, idx2 = prepare_probe2_data(features, categories, question_indices)
    
    print(f"\n📊 Probe 2 Data:")
    print(f"   Total samples: {len(y2)}")
    print(f"   Distracted by hint (+1): {sum(y2 == 1)}")
    print(f"   Found answer (-1): {sum(y2 == -1)}")
    
    if len(y2) < 10:
        print("⚠️  Not enough samples for Probe 2. Skipping...")
        probe2_results = None
    elif len(set(y2)) < 2:
        print("⚠️  Only one class present in Probe 2 data. Skipping...")
        probe2_results = None
    else:
        probe2_results = train_linear_probe(X2, y2, test_size, random_state)
        
        probe2_labels = {
            1: "distracted_by_hint (hit token limit)",
            -1: "found_answer (answered within budget)"
        }
        print_probe_results(probe2_results, "Distracted vs Found Answer", probe2_labels)
        
        if save_results:
            import os
            os.makedirs(output_dir, exist_ok=True)
            save_probe_results(
                probe2_results,
                "distracted_vs_found_answer",
                f"{output_dir}/probe2_distracted_vs_found_layer{layer}_pos{position}.json",
                layer, position, probe2_labels
            )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"   Layer: {layer}")
    print(f"   Position: {position}")
    print(f"   Feature dimension: {features.shape[1]}")
    
    print(f"\n📈 Results:")
    if probe1_results:
        print(f"   Probe 1 (Agrees vs Against): {probe1_results.accuracy*100:.2f}% accuracy")
    else:
        print(f"   Probe 1 (Agrees vs Against): Skipped (insufficient data)")
    
    if probe2_results:
        print(f"   Probe 2 (Distracted vs Found): {probe2_results.accuracy*100:.2f}% accuracy")
    else:
        print(f"   Probe 2 (Distracted vs Found): Skipped (insufficient data)")
    
    print("="*80)
    
    return probe1_results, probe2_results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on residual stream hidden states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with auto-detected layer and position
  python linear_probe_training.py hidden_values/labeled_results.json

  # Specify layer and position
  python linear_probe_training.py hidden_values/labeled_results.json --layer 1 --position 114

  # Change test split and random seed
  python linear_probe_training.py hidden_values/labeled_results.json --test-size 0.3 --seed 123

Expected JSON structure:
  The input JSON should have a "results" array where each element has:
    - "category": one of "correct_answer", "incorrect_not_hint", "chose_hint", "exceeded_budget"
    - "residual_stream_hidden_states": {"layer_1": {"114": [...]}, ...}
        """
    )
    
    parser.add_argument('json_path', type=str,
                        help='Path to labeled JSON file with residual stream data')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer to extract features from (auto-detected if not specified)')
    parser.add_argument('--position', type=int, default=None,
                        help='Token position to extract features from (auto-detected if not specified)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='probe_results',
                        help='Directory to save results (default: probe_results)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    
    args = parser.parse_args()
    
    try:
        run_experiments(
            json_path=args.json_path,
            layer=args.layer,
            position=args.position,
            test_size=args.test_size,
            random_state=args.seed,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

