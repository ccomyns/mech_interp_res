#!/usr/bin/env python3
"""
Linear Probe Training for Residual Stream Analysis

This script trains a binary linear probe on residual stream hidden states extracted
from Gemma-3-4B's responses to MMLU questions with incorrect hints.

The probe classifies:
    - correct_answer (-1): Model chose the correct answer
    - agrees_with_hint (+1): Model agreed with the incorrect hint

Each question has exactly one token position saved per layer, so the position is
auto-detected for each question.

Usage:
    python linear_probe_training.py <categorized_json_path> <bulk_file1> <bulk_file2> [options]

Example:
    python linear_probe_training.py data_collection_linear_probe/categorized_data.json \
        data_collection_linear_probe/bulk_residual_stream_incorrect_hint_q0-569_tokens1_20251202_214545.json \
        data_collection_linear_probe/bulk_residual_stream_incorrect_hint_q570-1139_tokens1_20251203_210144.json \
        --layer 1
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

def load_categorized_data(json_path: str) -> Tuple[Dict[str, Any], Dict[str, List[Dict]]]:
    """
    Load categorized data from JSON file.
    
    Expected JSON structure:
    {
        "metadata": {...},
        "statistics": {...},
        "correct_answer": [
            {"question_idx": 5, "answer_letter": "B", "correct_answer_letter": "B", 
             "incorrect_hint": "A", "split": "train"},
            ...
        ],
        "agrees_with_hint": [
            {"question_idx": 2, "answer_letter": "D", "correct_answer_letter": "B", 
             "incorrect_hint": "D", "split": "train"},
            ...
        ]
    }
    
    Returns:
        Tuple of (metadata, categorized_data_dict)
    """
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
    print(f"\n📊 Statistics:")
    for key, value in statistics.items():
        print(f"   {key}: {value}")
    
    return metadata, categorized_data


def load_bulk_residual_streams(bulk_files: List[str], 
                                question_indices: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Load residual stream data for specific question indices from bulk JSON files.
    
    Args:
        bulk_files: List of paths to bulk residual stream JSON files
        question_indices: List of question indices to load
        
    Returns:
        Dictionary mapping question_idx to residual stream data
    """
    print(f"\n📂 Loading residual streams from {len(bulk_files)} bulk file(s)...")
    
    # Create a set for faster lookup
    target_indices = set(question_indices)
    residual_streams = {}
    
    for bulk_file in bulk_files:
        print(f"   Loading: {bulk_file}")
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
        print(f"   Missing indices: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
    
    return residual_streams


def extract_features_from_categorized(categorized_data: Dict[str, List[Dict]],
                                      residual_streams: Dict[int, Dict[str, Any]],
                     layer: int, 
                                      split: str = 'train') -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract feature vectors and labels from categorized data and residual streams.
    
    Each question has exactly one position saved per layer, so position is auto-detected
    for each question.
    
    Args:
        categorized_data: Dictionary with 'correct_answer' and 'agrees_with_hint' lists
        residual_streams: Dictionary mapping question_idx to residual stream data
        layer: Which layer to extract (e.g., 1 or 18)
        split: Which split to extract ('train' or 'test')
        
    Returns:
        Tuple of (features array, labels array, question indices list)
    """
    features = []
    labels = []
    question_indices = []
    
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
        
        # Get the single position available for this question
        layer_data = hidden_states[layer_key]
        if not layer_data:
            continue
        
        # There should be exactly one position - get it
        position_key = list(layer_data.keys())[0]
        hidden_vector = np.array(layer_data[position_key])
        features.append(hidden_vector)
        labels.append(-1)  # correct_answer
        question_indices.append(q_idx)
    
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
        
        # Get the single position available for this question
        layer_data = hidden_states[layer_key]
        if not layer_data:
            continue
        
        # There should be exactly one position - get it
        position_key = list(layer_data.keys())[0]
        hidden_vector = np.array(layer_data[position_key])
        features.append(hidden_vector)
        labels.append(+1)  # agrees_with_hint
        question_indices.append(q_idx)
    
    return np.array(features), np.array(labels), question_indices


def get_available_layers(residual_streams: Dict[int, Dict[str, Any]]) -> List[int]:
    """Get the available layers from the residual stream data."""
    layers = set()
    
    for q_idx, data in residual_streams.items():
        hidden_states = data.get('residual_stream_hidden_states', {})
        for layer_key in hidden_states.keys():
            if layer_key.startswith('layer_'):
                layers.add(int(layer_key.split('_')[1]))
    
    return sorted(layers)


# ============================================================================
# Probe Training Functions
# ============================================================================

def train_linear_probe(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       random_state: int = 42,
                       n_cv_folds: int = 5,
                       regularization: float = 1.0) -> ProbeResults:
    """
    Train a linear probe (logistic regression) on the data.
    
    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training labels (+1 or -1)
        X_test: Test feature matrix
        y_test: Test labels
        random_state: Random seed for reproducibility
        n_cv_folds: Number of cross-validation folds
        regularization: L2 regularization strength (C parameter)
        
    Returns:
        ProbeResults object with training results
    """
    # Check if we have enough samples
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
        solver='lbfgs',
        random_state=random_state
    )
    probe.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = probe.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation on training set
    cv_scores = cross_val_score(probe, X_train_scaled, y_train, cv=min(n_cv_folds, n_samples // 2))
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Correct Answer (-1)', 'Agrees with Hint (+1)'])
    
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
        feature_dim=X_train.shape[1],
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
                       position: Optional[int],
                       label_descriptions: Dict[int, str]):
    """Save probe results to JSON file."""
    output_data = {
        'probe_name': probe_name,
        'layer': layer,
        'position': position,  # May be None if auto-detected per question
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

def run_experiment(categorized_json_path: str,
                   bulk_files: List[str],
                    layer: Optional[int] = None,
                    random_state: int = 42,
                    save_results: bool = True,
                    output_dir: str = "probe_results"):
    """
    Run linear probe experiment: Correct Answer vs Agrees with Hint.
    
    Args:
        categorized_json_path: Path to categorized JSON data
        bulk_files: List of paths to bulk residual stream JSON files
        layer: Layer to extract features from (auto-detected if None)
        random_state: Random seed
        save_results: Whether to save results to files
        output_dir: Directory for output files
    """
    print("="*80)
    print("LINEAR PROBE TRAINING: Correct Answer vs Agrees with Hint")
    print("="*80)
    
    # Load categorized data
    metadata, categorized_data = load_categorized_data(categorized_json_path)
    
    # Get all question indices from both categories
    all_indices = []
    for item in categorized_data['correct_answer']:
        all_indices.append(item['question_idx'])
    for item in categorized_data['agrees_with_hint']:
        all_indices.append(item['question_idx'])
    
    # Load residual streams for those indices
    residual_streams = load_bulk_residual_streams(bulk_files, all_indices)
    
    # Auto-detect layer if not specified
    available_layers = get_available_layers(residual_streams)
    print(f"\n📍 Available layers: {available_layers}")
    
    if layer is None:
        layer = available_layers[0] if available_layers else 1
        print(f"   Using layer: {layer} (auto-selected)")
    
    # Extract training features
    print(f"\n🔄 Extracting training features from layer {layer}...")
    X_train, y_train, train_indices = extract_features_from_categorized(
        categorized_data, residual_streams, layer, split='train'
    )
    
    if len(X_train) == 0:
        raise ValueError("No valid training features extracted. Check layer setting.")
    
    print(f"✅ Extracted {len(X_train)} training feature vectors of dimension {X_train.shape[1]}")
    print(f"   Correct answer (-1): {sum(y_train == -1)}")
    print(f"   Agrees with hint (+1): {sum(y_train == 1)}")
    
    # Extract test features
    print(f"\n🔄 Extracting test features from layer {layer}...")
    X_test, y_test, test_indices = extract_features_from_categorized(
        categorized_data, residual_streams, layer, split='test'
    )
    
    if len(X_test) == 0:
        raise ValueError("No valid test features extracted. Check layer setting.")
    
    print(f"✅ Extracted {len(X_test)} test feature vectors of dimension {X_test.shape[1]}")
    print(f"   Correct answer (-1): {sum(y_test == -1)}")
    print(f"   Agrees with hint (+1): {sum(y_test == 1)}")
    
    # Train probe
    print("\n" + "="*80)
    print("TRAINING LINEAR PROBE")
    print("="*80)
    
    probe_results = train_linear_probe(X_train, y_train, X_test, y_test, random_state)
        
    probe_labels = {
        -1: "correct_answer (chose the correct answer)",
        1: "agrees_with_hint (agreed with incorrect hint)"
        }
    print_probe_results(probe_results, "Correct Answer vs Agrees with Hint", probe_labels)
        
    if save_results:
        import os
        os.makedirs(output_dir, exist_ok=True)
        save_probe_results(
            probe_results,
            "correct_vs_agrees_with_hint",
            f"{output_dir}/probe_correct_vs_hint_layer{layer}_manual_review.json",
            layer, None, probe_labels
        )
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"   Layer: {layer}")
    print(f"   Feature dimension: {X_train.shape[1]}")
    
    print(f"\n📈 Results:")
    print(f"   Test Accuracy: {probe_results.accuracy*100:.2f}%")
    print(f"   Cross-Val Accuracy: {probe_results.cross_val_accuracy*100:.2f}% ± {probe_results.cross_val_std*100:.2f}%")
    
    print("="*80)
    
    return probe_results


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train linear probe on residual stream hidden states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with two bulk files
  python linear_probe_training.py categorized_data.json bulk1.json bulk2.json

  # Specify layer
  python linear_probe_training.py categorized_data.json bulk1.json bulk2.json --layer 1

  # Change random seed
  python linear_probe_training.py categorized_data.json bulk1.json bulk2.json --seed 123

Expected JSON structures:
  Categorized JSON:
    {
      "correct_answer": [{"question_idx": 5, "split": "train"}, ...],
      "agrees_with_hint": [{"question_idx": 2, "split": "train"}, ...]
    }
  
  Bulk JSON (each question has one token position saved per layer):
    {
      "results": [
        {
          "question_idx": 0,
          "residual_stream_hidden_states": {"layer_1": {"143": [...]}, "layer_18": {"143": [...]}}
        },
        ...
      ]
    }
        """
    )
    
    parser.add_argument('categorized_json', type=str,
                        help='Path to categorized JSON file with question indices and labels')
    parser.add_argument('bulk_files', type=str, nargs='+',
                        help='Paths to bulk residual stream JSON files')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer to extract features from (auto-detected if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='probe_results',
                        help='Directory to save results (default: probe_results)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    
    args = parser.parse_args()
    
    try:
        run_experiment(
            categorized_json_path=args.categorized_json,
            bulk_files=args.bulk_files,
            layer=args.layer,
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

