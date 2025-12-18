#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Probe Inference on Misleading Cognitive Questions

Uses the combined linear probe (trained on MMLU incorrect hint modes) to
classify residual stream activations from misleading cognitive questions.

The probe has two classes:
- "correct_answer" (positive direction): model answers correctly
- "agrees_with_hint" (negative direction): model follows the misleading distractor

Maps:
- is_correct = True  → expected class "correct_answer" (positive)
- is_correct = False → expected class "agrees_with_hint" (negative)
"""

import json
import numpy as np
from pathlib import Path


def load_probe(probe_path: str) -> tuple:
    """Load linear probe weights and bias."""
    with open(probe_path, 'r') as f:
        probe_data = json.load(f)
    
    weights = np.array(probe_data['probe_weights'])
    bias = probe_data['probe_bias']
    
    return weights, bias, probe_data.get('metadata', {})


def load_inference_results(results_path: str) -> tuple:
    """Load inference results with residual stream values."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data['results'], data.get('metadata', {}), data.get('summary', {})


def extract_hidden_state(result: dict) -> np.ndarray:
    """Extract layer 18 hidden state from a result entry."""
    hidden_states = result['residual_stream_hidden_states']['layer_18']
    # Get the first (and likely only) token position
    token_pos = list(hidden_states.keys())[0]
    return np.array(hidden_states[token_pos])


def predict_class(hidden_state: np.ndarray, weights: np.ndarray, bias: float) -> tuple:
    """
    Predict class using linear probe.
    
    Returns:
        (prediction, logit) where:
        - prediction: +1 for "correct_answer", -1 for "agrees_with_hint"
        - logit: raw score (positive = correct_answer direction)
    """
    logit = np.dot(hidden_state, weights) + bias
    prediction = 1 if logit > 0 else -1
    return prediction, logit


def run_inference(probe_path: str, results_path: str):
    """Run linear probe inference on all results."""
    
    # Load probe
    weights, bias, probe_meta = load_probe(probe_path)
    print(f"Loaded probe from: {probe_path}")
    print(f"  - Feature dim: {len(weights)}")
    print(f"  - Training accuracy: {probe_meta.get('layer', 'N/A')}")
    
    # Load results
    results, results_meta, summary = load_inference_results(results_path)
    print(f"\nLoaded results from: {results_path}")
    print(f"  - Total questions: {len(results)}")
    print(f"  - Model accuracy: {summary.get('accuracy', 'N/A')}")
    
    # Run inference
    print("\n" + "=" * 80)
    print("PROBE INFERENCE RESULTS")
    print("=" * 80)
    
    correct_predictions = 0
    total = 0
    
    # Track by ground truth class
    correct_when_model_correct = 0
    total_model_correct = 0
    correct_when_model_incorrect = 0
    total_model_incorrect = 0
    
    detailed_results = []
    
    for result in results:
        if 'residual_stream_hidden_states' not in result:
            continue
        
        hidden_state = extract_hidden_state(result)
        prediction, logit = predict_class(hidden_state, weights, bias)
        
        # Ground truth based on is_correct
        is_correct = result['is_correct']
        expected = 1 if is_correct else -1  # +1 = correct_answer, -1 = agrees_with_hint
        
        probe_correct = (prediction == expected)
        correct_predictions += int(probe_correct)
        total += 1
        
        # Track by class
        if is_correct:
            total_model_correct += 1
            correct_when_model_correct += int(probe_correct)
        else:
            total_model_incorrect += 1
            correct_when_model_incorrect += int(probe_correct)
        
        detailed_results.append({
            'question_idx': result['question_idx'],
            'is_correct': is_correct,
            'expected_class': 'correct_answer' if expected == 1 else 'agrees_with_hint',
            'predicted_class': 'correct_answer' if prediction == 1 else 'agrees_with_hint',
            'logit': logit,
            'probe_correct': probe_correct
        })
    
    # Print summary
    accuracy = correct_predictions / total if total > 0 else 0
    print(f"\nOverall Probe Accuracy: {correct_predictions}/{total} = {accuracy:.2%}")
    
    if total_model_correct > 0:
        acc_correct = correct_when_model_correct / total_model_correct
        print(f"  - When model was correct (expect 'correct_answer'): {correct_when_model_correct}/{total_model_correct} = {acc_correct:.2%}")
    
    if total_model_incorrect > 0:
        acc_incorrect = correct_when_model_incorrect / total_model_incorrect
        print(f"  - When model was incorrect (expect 'agrees_with_hint'): {correct_when_model_incorrect}/{total_model_incorrect} = {acc_incorrect:.2%}")
    
    # Print detailed results
    print("\n" + "-" * 80)
    print("DETAILED RESULTS")
    print("-" * 80)
    print(f"{'Idx':>4} | {'Model':>8} | {'Expected':>15} | {'Predicted':>15} | {'Logit':>10} | {'Match':>5}")
    print("-" * 80)
    
    for r in detailed_results:
        model_result = "correct" if r['is_correct'] else "wrong"
        match = "Y" if r['probe_correct'] else "N"
        print(f"{r['question_idx']:>4} | {model_result:>8} | {r['expected_class']:>15} | {r['predicted_class']:>15} | {r['logit']:>10.4f} | {match:>5}")
    
    # Distribution of logits
    print("\n" + "-" * 80)
    print("LOGIT DISTRIBUTION")
    print("-" * 80)
    
    logits_correct = [r['logit'] for r in detailed_results if r['is_correct']]
    logits_incorrect = [r['logit'] for r in detailed_results if not r['is_correct']]
    
    if logits_correct:
        print(f"When model correct (should be positive):   mean={np.mean(logits_correct):>8.4f}, std={np.std(logits_correct):>8.4f}")
    if logits_incorrect:
        print(f"When model incorrect (should be negative): mean={np.mean(logits_incorrect):>8.4f}, std={np.std(logits_incorrect):>8.4f}")
    
    return detailed_results


def main():
    # Paths
    script_dir = Path(__file__).parent
    probe_path = script_dir.parent / "probe_results" / "combined_probe_layer18.json"
    results_path = script_dir / "misleading_cognitive_inference_q0-49_tokens1_20251218_120620.json"
    
    print("=" * 80)
    print("LINEAR PROBE INFERENCE ON MISLEADING COGNITIVE QUESTIONS")
    print("=" * 80)
    
    run_inference(str(probe_path), str(results_path))


if __name__ == "__main__":
    main()

