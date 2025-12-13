import argparse
import os
import json
import subprocess
import sys

def run_evaluation(model_dir, gt_dir, output_file, model_name):
    """Run the evaluation.py script for a given model output."""
    cmd = [
        sys.executable, "evaluation.py",
        "--model_output", model_dir,
        "--ground_truth", gt_dir,
        "--output", output_file,
        "--model_name", model_name
    ]
    print(f"Running evaluation for {model_name}...")
    subprocess.check_call(cmd)

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Compare VLM prompts')
    parser.add_argument('--enhanced', required=True, help='Directory for enhanced prompt outputs')
    parser.add_argument('--standard', required=True, help='Directory for standard prompt outputs')
    parser.add_argument('--ground_truth', required=True, help='Directory for ground truth')
    parser.add_argument('--output_dir', required=True, help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    enhanced_out = os.path.join(args.output_dir, "evaluation_enhanced.json")
    standard_out = os.path.join(args.output_dir, "evaluation_standard.json")
    comparison_out = os.path.join(args.output_dir, "comparison_results.json")
    
    # Run evaluation for Enhanced Prompt
    run_evaluation(args.enhanced, args.ground_truth, enhanced_out, "Enhanced Prompt")
    
    # Run evaluation for Standard Prompt
    run_evaluation(args.standard, args.ground_truth, standard_out, "Standard Prompt")
    
    # Load and Compare
    res_enhanced = load_results(enhanced_out)
    res_standard = load_results(standard_out)
    
    # Extract 0.7 threshold results
    metrics_enhanced = res_enhanced['results']['0.7']['aggregate']
    metrics_standard = res_standard['results']['0.7']['aggregate']
    
    comparison = {
        "enhanced": metrics_enhanced,
        "standard": metrics_standard,
        "improvement": {
            "entities_f1": metrics_enhanced['entities']['f1'] - metrics_standard['entities']['f1'],
            "relations_f1": metrics_enhanced['relations']['f1'] - metrics_standard['relations']['f1']
        }
    }
    
    with open(comparison_out, 'w') as f:
        json.dump(comparison, f, indent=2)
        
    print(f"\nComparison Complete! Results saved to {args.output_dir}")
    print(f"Enhanced Entity F1: {metrics_enhanced['entities']['f1']:.4f}")
    print(f"Standard Entity F1: {metrics_standard['entities']['f1']:.4f}")
    print(f"Improvement: {comparison['improvement']['entities_f1']:.4f}")

if __name__ == "__main__":
    main()
