#!/usr/bin/env python3
"""
Component Glossary Generator

Analyzes processed PDFs to suggest components.json updates using
frequency analysis, lemmatization, and POS tagging.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import spacy
    nlp = spacy.load("nl_core_news_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: SpaCy not available. Install with: python -m spacy download nl_core_news_sm")
    sys.exit(1)


def analyze_components_from_graphs(graph_dir: Path) -> Dict[str, Dict]:
    """
    Analyze all processed graphs to extract component candidates.
    
    Returns:
        Dict mapping lemmatized component -> {
            'forms': [list of surface forms],
            'frequency': total_count,
            'confidence': calculated_confidence
        }
    """
    if not SPACY_AVAILABLE:
        return {}
    
    component_candidates = defaultdict(lambda: {'forms': set(), 'frequency': 0, 'mentions': []})
    
    # Common words to exclude
    common_words = {
        "de", "het", "een", "van", "op", "in", "voor", "met", "aan", "bij", "naar",
        "is", "zijn", "werkt", "wordt", "hebben", "kan", "moet", "zou",
        "correct", "goed", "klaar", "ok", "nee", "ja", "periode", "auteur",
        "versiedatum", "opmerkingen", "laatste", "wijzigingen", "wissel",
        "vervang", "stel", "maak", "controleer", "reinig", "afstellen"
    }
    
    # Adjectives and states to exclude
    adjectives_to_exclude = {
        "statisch", "vervuild", "versleten", "defect", "correct", "goed",
        "vervangen", "schoon", "groot", "klein", "lang", "kort"
    }
    
    # Process all graph files
    graph_files = list(graph_dir.glob("semantic_graph_*.json"))
    if not graph_files:
        print(f"No graph files found in {graph_dir}")
        return {}
    
    print(f"Analyzing {len(graph_files)} graph file(s)...")
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {graph_file}: {e}")
            continue
        
        # Get all text from Action, Condition, Observation nodes
        all_text = ' '.join([
            n.get('label', '') 
            for n in graph['nodes'] 
            if n.get('type') in ['Action', 'Condition', 'Observation']
        ])
        
        if not all_text.strip():
            continue
        
        # Process with SpaCy
        doc = nlp(all_text)
        
        # Extract noun phrases and lemmatize
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            chunk_lower = chunk_text.lower()
            
            # Filter criteria (same as in extract_entities)
            if len(chunk_text) <= 4:
                continue
            
            # Skip common words
            if chunk_lower in common_words:
                continue
            
            # Skip adjectives/states
            if chunk_lower in adjectives_to_exclude:
                continue
            
            # Skip if contains verbs
            if any(token.pos_ == "VERB" for token in chunk):
                continue
            
            # Must contain at least one noun
            if not any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
                continue
            
            # Skip document references
            if re.match(r'^[A-Z]{2}-\d+-\d+-\d+', chunk_text):
                continue
            
            # Skip coordinate references
            if re.match(r'^[xyz][/\\][xyz]', chunk_text, re.IGNORECASE):
                continue
            
            # Get lemmatized form (use nouns as base)
            nouns = [token.lemma_.lower() for token in chunk if token.pos_ in ["NOUN", "PROPN"]]
            if not nouns:
                continue
            
            # For multi-word phrases, use all nouns; for single noun, use it
            if len(nouns) == 1:
                canonical = nouns[0]
            else:
                # For phrases like "de vacuumkopjes", use the main noun
                # Remove articles and determiners
                articles = {"de", "het", "een", "van", "op", "in", "voor", "met", "aan"}
                filtered_nouns = [n for n in nouns if n not in articles]
                canonical = ' '.join(filtered_nouns) if filtered_nouns else nouns[0]
            
            # Skip generic terms that are not domain-specific components
            generic_terms = {
                "midden", "positie", "oppakken", "plaatsen", "onderdeel", "onderdelen",
                "waarde", "stand", "processpecificatieblad", "processpecificatie",
                "abc", "cf", "pressto", "dp17", "dp71", "dp81", "dp91", "dp52", "dp61"
            }
            if canonical in generic_terms or any(gt in canonical for gt in generic_terms):
                continue
            
            # Store surface form and increment frequency
            component_candidates[canonical]['forms'].add(chunk_text.lower())
            component_candidates[canonical]['frequency'] += 1
            component_candidates[canonical]['mentions'].append({
                'text': chunk_text,
                'file': graph_file.name
            })
    
    # Convert sets to lists and calculate confidence
    result = {}
    for canonical, data in component_candidates.items():
        frequency = data['frequency']
        form_count = len(data['forms'])
        
        # Higher confidence if:
        # - High frequency (>= 3 mentions)
        # - Consistent form (few variations)
        # - Single word (more reliable than phrases)
        is_single_word = len(canonical.split()) == 1
        confidence = 0.5 + (min(frequency, 10) * 0.05) + (min(form_count, 3) * 0.05) + (0.1 if is_single_word else 0)
        confidence = min(0.95, confidence)
        
        result[canonical] = {
            'forms': sorted(list(data['forms'])),
            'frequency': frequency,
            'confidence': round(confidence, 2),
            'mentions': data['mentions'][:5]  # Sample mentions
        }
    
    return result


def generate_glossary_suggestions(
    current_glossary: List[str],
    analyzed_components: Dict[str, Dict],
    min_frequency: int = 2,
    min_confidence: float = 0.6
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Generate suggestions for glossary updates.
    
    Returns:
        (to_add, to_remove, to_keep)
    """
    current_lower = {c.lower().strip() for c in current_glossary}
    
    # Components to add (high frequency, not in current glossary)
    to_add = []
    for canonical, data in analyzed_components.items():
        if (data['frequency'] >= min_frequency and 
            data['confidence'] >= min_confidence):
            
            # Check if already in glossary (exact match or variant)
            already_in = False
            for term in current_lower:
                # Check if canonical matches or is variant of existing term
                if (canonical == term or 
                    canonical in term or 
                    term in canonical or
                    any(form == term for form in data['forms'])):
                    already_in = True
                    break
            
            if not already_in:
                # Use most common surface form, but normalize (remove articles)
                most_common_form = data['forms'][0] if data['forms'] else canonical
                
                # Normalize: remove leading articles
                normalized_form = most_common_form
                words = normalized_form.split()
                while words and words[0] in {"de", "het", "een"}:
                    words = words[1:]
                if words:
                    normalized_form = ' '.join(words)
                
                # Skip if normalized form is too generic
                generic_terms = {
                    "midden", "positie", "oppakken", "plaatsen", "onderdeel", "onderdelen",
                    "waarde", "stand", "processpecificatieblad", "processpecificatie"
                }
                if normalized_form in generic_terms or len(normalized_form) < 4:
                    continue
                
                to_add.append({
                    'term': normalized_form,
                    'canonical': canonical,
                    'frequency': data['frequency'],
                    'confidence': data['confidence'],
                    'variants': data['forms']
                })
    
    # Components to remove (in glossary but not found in PDFs)
    to_remove = []
    for term in current_glossary:
        term_lower = term.lower().strip()
        found = False
        
        # Check if term appears in analyzed components
        for canonical, data in analyzed_components.items():
            if (term_lower == canonical or
                term_lower in canonical or
                canonical in term_lower or
                any(term_lower == form or term_lower in form or form in term_lower 
                    for form in data['forms'])):
                found = True
                break
        
        if not found:
            to_remove.append(term)
    
    # Components to keep (found in both)
    to_keep = []
    for term in current_glossary:
        term_lower = term.lower().strip()
        found = any(
            term_lower == canonical or
            term_lower in canonical or
            canonical in term_lower or
            any(term_lower == form or term_lower in form or form in term_lower 
                for form in data['forms'])
            for canonical, data in analyzed_components.items()
        )
        if found:
            to_keep.append(term)
    
    return to_add, to_remove, to_keep


def main():
    """Main function to analyze and suggest glossary updates."""
    project_root = Path(__file__).parent.parent
    graph_dir = project_root / "data" / "processed"
    glossary_path = project_root / "data" / "glossary" / "components.json"
    
    if not SPACY_AVAILABLE:
        print("ERROR: SpaCy not available. Cannot perform analysis.")
        sys.exit(1)
    
    # Load current glossary
    if not glossary_path.exists():
        print(f"ERROR: Glossary not found at {glossary_path}")
        sys.exit(1)
    
    with open(glossary_path, 'r', encoding='utf-8') as f:
        current_glossary = json.load(f)
    
    print("=" * 60)
    print("Component Glossary Generator")
    print("=" * 60)
    print(f"Analyzing processed graphs in: {graph_dir}")
    print(f"Current glossary: {len(current_glossary)} components")
    
    analyzed = analyze_components_from_graphs(graph_dir)
    
    if not analyzed:
        print("No components found. Make sure you have processed at least one PDF.")
        sys.exit(1)
    
    print(f"\nFound {len(analyzed)} unique component candidates")
    
    # Generate suggestions
    to_add, to_remove, to_keep = generate_glossary_suggestions(
        current_glossary, analyzed, min_frequency=2, min_confidence=0.6
    )
    
    print(f"\n{'='*60}")
    print("GLOSSARY UPDATE SUGGESTIONS")
    print(f"{'='*60}")
    
    print(f"\n📝 TO ADD ({len(to_add)} components):")
    for item in sorted(to_add, key=lambda x: x['frequency'], reverse=True):
        print(f"  + {item['term']:35} (freq: {item['frequency']:2}, conf: {item['confidence']:.2f})")
        if len(item['variants']) > 1:
            print(f"    Variants: {', '.join(item['variants'][:3])}")
    
    print(f"\n❌ TO REMOVE ({len(to_remove)} components - not found in PDFs):")
    for term in sorted(to_remove):
        print(f"  - {term}")
    
    print(f"\n✅ TO KEEP ({len(to_keep)} components - found in PDFs):")
    for term in sorted(to_keep):
        print(f"  ✓ {term}")
    
    # Generate updated glossary
    # Keep existing terms that were found, add new high-confidence terms
    updated_glossary = sorted(set(to_keep + [item['term'] for item in to_add]))
    
    print(f"\n{'='*60}")
    print(f"UPDATED GLOSSARY ({len(updated_glossary)} components)")
    print(f"{'='*60}")
    
    # Save suggestions
    suggestions_file = project_root / "data" / "glossary" / "glossary_suggestions.json"
    suggestions_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(suggestions_file, 'w', encoding='utf-8') as f:
        json.dump({
            'to_add': to_add,
            'to_remove': to_remove,
            'to_keep': to_keep,
            'updated_glossary': updated_glossary,
            'analysis_stats': {
                'total_candidates': len(analyzed),
                'high_frequency': len([c for c in analyzed.values() if c['frequency'] >= 3]),
                'current_glossary_size': len(current_glossary),
                'suggested_size': len(updated_glossary)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuggestions saved to: {suggestions_file}")
    
    # Ask if user wants to apply
    if '--apply' in sys.argv:
        print("\nApplying updates to components.json...")
        with open(glossary_path, 'w', encoding='utf-8') as f:
            json.dump(updated_glossary, f, indent=2, ensure_ascii=False)
        print(f"✓ Updated {glossary_path}")
    else:
        print("\nReview the suggestions and update components.json manually, or")
        print("run with --apply flag to auto-update (after review).")
        print(f"\nTo apply: python {Path(__file__).name} --apply")


if __name__ == "__main__":
    main()

