import json
from pathlib import Path
from collections import defaultdict

from backend.ingestion.obligation_extractor import ObligationExtractor, Obligation, load_obligations, save_obligations
from backend.ingestion.chunker import RegulationChunk

def main():
    root_dir = Path(__file__).parent.parent
    
    # 1. Load obligations
    obligations = load_obligations("eu_ai_act")
    extracted_articles = set(obl.article for obl in obligations if obl.article)
    
    # 3. Load chunks
    chunks_path = root_dir / "data" / "processed" / "eu_ai_act_chunks.json"
    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}")
        return
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    chunks = [RegulationChunk.from_dict(item) for item in chunks_data]
    
    # Build a map of article -> chunks
    article_to_chunks = defaultdict(list)
    all_articles = set()
    for chunk in chunks:
        if chunk.article:
            all_articles.add(chunk.article)
            article_to_chunks[chunk.article].append(chunk)
            
    # 2. Identify missing articles
    missing_articles = all_articles - extracted_articles
    
    if not missing_articles:
        print("No articles are missing obligations.")
        return
        
    print(f"Found {len(missing_articles)} articles with zero obligations.")
    
    # 4. Retry extraction
    extractor = ObligationExtractor()
    recovered_obligations = []
    recovered_articles = []
    
    for article in sorted(missing_articles):
        article_chunks = article_to_chunks[article]
        try:
            print(f"Retrying Article {article}...")
            new_obls = extractor.extract_by_article(
                "eu_ai_act",
                "EU AI Act",
                article,
                article_chunks
            )
            
            if new_obls:
                recovered_obligations.extend(new_obls)
                recovered_articles.append(article)
                print(f"  -> Recovered {len(new_obls)} obligations.")
            else:
                print(f"  -> Still 0 obligations.")
                
        except Exception as e:
            print(f"  -> Error on Article {article}: {e}")
            
    # 5. Append new obligations
    if recovered_obligations:
        obligations.extend(recovered_obligations)
        save_obligations(obligations, "eu_ai_act")
        
    # 6. Print which articles were recovered
    print(f"\nRecovered obligations for these articles: {', '.join(sorted(recovered_articles))}")
    print(f"Total new obligations added: {len(recovered_obligations)}")

if __name__ == "__main__":
    main()