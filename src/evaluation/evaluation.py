import json
import os
import argparse
import re
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import spacy
from fuzzywuzzy import fuzz
from statistics import mean

# Load spaCy model for Dutch
try:
    nlp = spacy.load("nl_core_news_sm")
except OSError:
    print("Downloading Dutch spaCy model...")
    os.system("python -m spacy download nl_core_news_sm")
    nlp = spacy.load("nl_core_news_sm")

class VLMEvaluator:
    def __init__(self, sensitivity_threshold: float = 0.70):
        self.threshold = sensitivity_threshold
        self.match_stats = defaultdict(int)

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing numbering, punctuation, and extra whitespace.
        Example: "4-0 Voldoet de kap?" -> "voldoet de kap"
        """
        if not text:
            return ""
        
        # Remove numbering patterns like "4-0", "A1)", "1a)", "19.1" at start
        text = re.sub(r'^[\w\d]+[-.)]\s*', '', text)
        text = re.sub(r'^\d+\.\d+\s*', '', text)
        
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        return ' '.join(text.split())

    def extract_keywords(self, text: str) -> Set[str]:
        """
        Extract meaningful keywords (nouns, verbs, adjectives) using spaCy.
        """
        doc = nlp(text)
        keywords = {
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop 
            and token.pos_ in ['NOUN', 'VERB', 'ADJ']
            and len(token.lemma_) > 2
        }
        return keywords

    def entities_match(self, e1: Dict, e2: Dict) -> Tuple[bool, float, str]:
        """
        Check if two entities match based on type and text.
        Returns: (matched, score, match_type)
        """
        # 1. Type check (Strict)
        if e1.get('type') != e2.get('type'):
            return False, 0.0, "type_mismatch"

        t1_raw = e1.get('text', '')
        t2_raw = e2.get('text', '')
        
        # Handle short text (Ja/Nee) strictly
        if len(t1_raw) < 5 or len(t2_raw) < 5:
            if t1_raw.lower().strip() == t2_raw.lower().strip():
                return True, 1.0, "exact_match_short"
            return False, 0.0, "mismatch_short"

        t1_norm = self.normalize_text(t1_raw)
        t2_norm = self.normalize_text(t2_raw)

        # 2. Exact Match (after normalization)
        if t1_norm == t2_norm:
            return True, 1.0, "exact_match"

        # 3. Substring Match
        if len(t1_norm) > 10 and len(t2_norm) > 10:
            if t1_norm in t2_norm or t2_norm in t1_norm:
                overlap = min(len(t1_norm), len(t2_norm)) / max(len(t1_norm), len(t2_norm))
                if overlap >= 0.6:
                    return True, overlap, "substring_match"

        # 4. Semantic Match (Fuzzy String)
        fuzz_ratio = fuzz.ratio(t1_norm, t2_norm) / 100.0
        if fuzz_ratio >= self.threshold:
            return True, fuzz_ratio, "semantic_match"

        # 5. Keyword Match
        k1 = self.extract_keywords(t1_raw)
        k2 = self.extract_keywords(t2_raw)
        
        if k1 and k2:
            intersection = k1.intersection(k2)
            union = k1.union(k2)
            jaccard = len(intersection) / len(union)
            
            if jaccard >= self.threshold:
                return True, jaccard, "keyword_match"

        return False, max(fuzz_ratio, 0.0), "no_match"

    def evaluate_entities(self, pred_entities: List[Dict], gt_entities: List[Dict]) -> Dict:
        """
        Evaluate entities using greedy matching.
        """
        tp = 0
        matched_gt = set()
        matched_pred = set()
        
        # Sort by length to match longer/more specific entities first (heuristic)
        # But for greedy matching, we might want to find best scores first.
        # Let's compute all pairwise scores.
        
        matches = []
        for i, p in enumerate(pred_entities):
            for j, g in enumerate(gt_entities):
                matched, score, method = self.entities_match(p, g)
                if matched:
                    matches.append({
                        'pred_idx': i,
                        'gt_idx': j,
                        'score': score,
                        'method': method
                    })
        
        # Sort matches by score descending
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        entity_map = {} # Map pred_id -> gt_id for relation evaluation
        
        for m in matches:
            if m['pred_idx'] not in matched_pred and m['gt_idx'] not in matched_gt:
                matched_pred.add(m['pred_idx'])
                matched_gt.add(m['gt_idx'])
                tp += 1
                self.match_stats[m['method']] += 1
                
                # Store mapping
                p_id = pred_entities[m['pred_idx']].get('id')
                g_id = gt_entities[m['gt_idx']].get('id')
                if p_id and g_id:
                    entity_map[p_id] = g_id

        fp = len(pred_entities) - tp
        fn = len(gt_entities) - tp
        
        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'entity_map': entity_map
        }

    def evaluate_relations(self, pred_relations: List[Dict], gt_relations: List[Dict], entity_map: Dict) -> Dict:
        """
        Evaluate relations based on matched entities.
        """
        tp = 0
        # Convert GT relations to a set of tuples for fast lookup: (source_gt_id, target_gt_id, type)
        gt_set = set()
        for r in gt_relations:
            gt_set.add((r['source'], r['target'], r['type']))
            
        matched_pred_indices = set()
        
        for i, r in enumerate(pred_relations):
            # Map predicted source/target IDs to GT IDs
            mapped_source = entity_map.get(r['source'])
            mapped_target = entity_map.get(r['target'])
            
            if mapped_source and mapped_target:
                if (mapped_source, mapped_target, r['type']) in gt_set:
                    tp += 1
                    matched_pred_indices.add(i)
                    
        fp = len(pred_relations) - tp
        fn = len(gt_relations) - tp
        
        return {'tp': tp, 'fp': fp, 'fn': fn}

    def flatten_model_output(self, model_data: List[Dict]) -> Dict:
        """
        Flatten list of pages into single entities/relations lists.
        """
        all_entities = []
        all_relations = []
        
        for page in model_data:
            if 'json' in page and page['json']:
                p_json = page['json']
                all_entities.extend(p_json.get('entities', []))
                all_relations.extend(p_json.get('relations', []))
                
        return {'entities': all_entities, 'relations': all_relations}

    def calculate_f1(self, tp: int, fp: int, fn: int) -> Dict:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM procedural knowledge extraction')
    parser.add_argument('--model_output', required=True, help='Directory containing model output JSONs')
    parser.add_argument('--ground_truth', required=True, help='Directory containing ground truth JSONs')
    parser.add_argument('--output', default='evaluation_results.json', help='Output JSON file')
    parser.add_argument('--model_name', default='VLM', help='Name of the model being evaluated')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    
    args = parser.parse_args()
    
    # Thresholds to test
    thresholds = [0.6, 0.7, 0.8, 0.9] if args.sensitivity else [0.7]
    
    all_results = {}
    
    for thresh in thresholds:
        evaluator = VLMEvaluator(sensitivity_threshold=thresh)
        
        total_metrics = {
            'entities': {'tp': 0, 'fp': 0, 'fn': 0},
            'relations': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        
        per_file_results = []
        
        # Iterate over ground truth files
        for filename in os.listdir(args.ground_truth):
            if not filename.endswith('.json'):
                continue
                
            gt_path = os.path.join(args.ground_truth, filename)
            
            # Find corresponding model output
            # Try exact name, or DP17- prefixed name
            model_filename = filename
            model_path = os.path.join(args.model_output, model_filename)
            
            if not os.path.exists(model_path):
                # Try with prefix
                model_filename = f"DP17-{filename.upper()}"
                model_path = os.path.join(args.model_output, model_filename)
                
                if not os.path.exists(model_path):
                     # Try with prefix but normal case
                    model_filename = f"DP17-{filename}"
                    model_path = os.path.join(args.model_output, model_filename)

            if not os.path.exists(model_path):
                print(f"Warning: No model output found for {filename}")
                continue
                
            # Load Data
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
                
            with open(model_path, 'r') as f:
                model_raw = json.load(f)
                
            # Preprocess Model Data (Flatten)
            if isinstance(model_raw, list):
                model_data = evaluator.flatten_model_output(model_raw)
            else:
                model_data = model_raw
                
            # Evaluate Entities
            e_metrics = evaluator.evaluate_entities(model_data.get('entities', []), gt_data.get('entities', []))
            
            # Evaluate Relations
            r_metrics = evaluator.evaluate_relations(model_data.get('relations', []), gt_data.get('relations', []), e_metrics['entity_map'])
            
            # Accumulate
            total_metrics['entities']['tp'] += e_metrics['tp']
            total_metrics['entities']['fp'] += e_metrics['fp']
            total_metrics['entities']['fn'] += e_metrics['fn']
            
            total_metrics['relations']['tp'] += r_metrics['tp']
            total_metrics['relations']['fp'] += r_metrics['fp']
            total_metrics['relations']['fn'] += r_metrics['fn']
            
            per_file_results.append({
                'file': filename,
                'entities': evaluator.calculate_f1(e_metrics['tp'], e_metrics['fp'], e_metrics['fn']),
                'relations': evaluator.calculate_f1(r_metrics['tp'], r_metrics['fp'], r_metrics['fn'])
            })
            
        # Calculate Aggregates
        agg_entities = evaluator.calculate_f1(total_metrics['entities']['tp'], total_metrics['entities']['fp'], total_metrics['entities']['fn'])
        agg_relations = evaluator.calculate_f1(total_metrics['relations']['tp'], total_metrics['relations']['fp'], total_metrics['relations']['fn'])
        
        all_results[str(thresh)] = {
            'aggregate': {
                'entities': agg_entities,
                'relations': agg_relations
            },
            'match_stats': dict(evaluator.match_stats),
            'per_file': per_file_results
        }
        
    # Output Results
    final_output = {
        'model_name': args.model_name,
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {args.output}")
    
    # Print Summary for 0.7 threshold
    res_07 = all_results.get('0.7', list(all_results.values())[0])
    print("\nAggregate Results (Threshold 0.7):")
    print(f"Entities: P={res_07['aggregate']['entities']['precision']:.2f}, R={res_07['aggregate']['entities']['recall']:.2f}, F1={res_07['aggregate']['entities']['f1']:.2f}")
    print(f"Relations: P={res_07['aggregate']['relations']['precision']:.2f}, R={res_07['aggregate']['relations']['recall']:.2f}, F1={res_07['aggregate']['relations']['f1']:.2f}")

if __name__ == "__main__":
    main()
