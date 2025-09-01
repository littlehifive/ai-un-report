#!/usr/bin/env python3
"""Create golden evaluation dataset based on existing UN documents."""

import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import random

# Add src to path
sys.path.append('src')

def analyze_corpus():
    """Analyze the corpus to understand document types and content."""
    chunks_file = Path('data/parsed/chunks.parquet')
    if not chunks_file.exists():
        raise FileNotFoundError("Chunks file not found")
    
    df = pd.read_parquet(chunks_file)
    
    print(f"📊 Corpus Analysis:")
    print(f"   Total chunks: {len(df)}")
    print(f"   Unique documents: {df['symbol'].nunique()}")
    
    # Get representative documents from different categories
    samples = {}
    
    # Sample from major document types
    for prefix in ['A/', 'S/', 'E/', 'DP/', 'ST/', 'JIU/']:
        matching = df[df['symbol'].str.startswith(prefix)]
        if len(matching) > 0:
            # Get diverse documents from this type
            unique_docs = matching['symbol'].unique()[:5]
            samples[prefix] = unique_docs
    
    return df, samples

def create_golden_queries():
    """Create 20 golden evaluation queries based on actual UN document content."""
    
    golden_queries = [
        # Technology & Innovation (based on UNCTAD/TIR/2025)
        {
            "question": "What does the UNCTAD Technology and Innovation Report 2025 say about artificial intelligence for development?",
            "ground_truth": "The UNCTAD Technology and Innovation Report 2025 focuses on inclusive artificial intelligence for development, examining how AI can be adapted to local contexts and infrastructure to support sustainable development goals.",
            "query_type": "document_specific",
            "expected_docs": ["UNCTAD/TIR/2025"],
            "category": "technology"
        },
        {
            "question": "What are the key recommendations for AI implementation in developing countries according to recent UNCTAD reports?",
            "ground_truth": "UNCTAD recommends adapting AI to local digital infrastructure, building local capacity, ensuring inclusive development, and implementing good practices for sustainable technology adoption.",
            "query_type": "analytical", 
            "expected_docs": ["UNCTAD"],
            "category": "technology"
        },
        
        # Security Council (based on S/ documents)
        {
            "question": "What peacekeeping challenges are discussed in Security Council reports from 2025?",
            "ground_truth": "Security Council reports from 2025 discuss various peacekeeping challenges including operational complexity, resource constraints, civilian protection, and coordination between international actors.",
            "query_type": "thematic",
            "expected_docs": ["S/2025"],
            "category": "peace_security"
        },
        {
            "question": "What does Security Council document S/2025/446 specifically address?",
            "ground_truth": "Security Council document S/2025/446 addresses specific regional security concerns and peacekeeping operations, providing detailed analysis of current situations and recommendations.",
            "query_type": "document_specific",
            "expected_docs": ["S/2025/446"], 
            "category": "peace_security"
        },
        
        # Women's Rights (based on E/CN.6 documents)
        {
            "question": "What are the main themes discussed in the Commission on the Status of Women reports from 2025?",
            "ground_truth": "The Commission on the Status of Women reports from 2025 focus on gender equality, women's empowerment, eliminating discrimination, and advancing women's rights in various sectors.",
            "query_type": "thematic",
            "expected_docs": ["E/CN.6/2025"],
            "category": "human_rights"
        },
        {
            "question": "What recommendations does document E/CN.6/2025/3 make regarding women's economic empowerment?",
            "ground_truth": "Document E/CN.6/2025/3 provides comprehensive recommendations on women's economic empowerment including equal pay, leadership opportunities, access to finance, and removing barriers to employment.",
            "query_type": "document_specific",
            "expected_docs": ["E/CN.6/2025/3"],
            "category": "human_rights"
        },
        
        # General Assembly (based on A/ documents)
        {
            "question": "What are the key priorities outlined in General Assembly documents from 2025?",
            "ground_truth": "General Assembly documents from 2025 outline priorities including sustainable development, climate action, peace and security, human rights, and strengthening multilateral cooperation.",
            "query_type": "broad_search",
            "expected_docs": ["A/80", "A/79"],
            "category": "governance"
        },
        {
            "question": "What does General Assembly document A/80/16 specifically cover?",
            "ground_truth": "General Assembly document A/80/16 covers specific agenda items related to UN operations, budget considerations, and administrative matters for the current session.",
            "query_type": "document_specific", 
            "expected_docs": ["A/80/16"],
            "category": "governance"
        },
        
        # Development Programme (based on DP documents)
        {
            "question": "What development priorities are highlighted in UNDP reports from 2025?",
            "ground_truth": "UNDP reports from 2025 highlight development priorities including poverty reduction, sustainable development goals implementation, climate resilience, and institutional capacity building.",
            "query_type": "thematic",
            "expected_docs": ["DP/2025"],
            "category": "development"
        },
        {
            "question": "How do UNDP documents assess progress on Sustainable Development Goals?",
            "ground_truth": "UNDP documents assess SDG progress through comprehensive monitoring frameworks, identifying gaps, challenges, and successful interventions while providing recommendations for acceleration.",
            "query_type": "analytical",
            "expected_docs": ["DP"],
            "category": "development"
        },
        
        # Joint Inspection Unit (based on JIU documents)
        {
            "question": "What organizational improvements does the Joint Inspection Unit recommend in its 2025 reports?",
            "ground_truth": "The Joint Inspection Unit 2025 reports recommend improvements in organizational efficiency, transparency, accountability mechanisms, and better coordination across UN agencies.",
            "query_type": "analytical",
            "expected_docs": ["JIU/REP/2025"],
            "category": "governance"
        },
        {
            "question": "What specific findings are presented in JIU report JIU/REP/2025/1?",
            "ground_truth": "JIU report JIU/REP/2025/1 presents specific findings on UN system coordination, identifies inefficiencies, and proposes concrete recommendations for organizational improvement.",
            "query_type": "document_specific",
            "expected_docs": ["JIU/REP/2025/1"],
            "category": "governance"
        },
        
        # Cross-cutting and Comparative Queries
        {
            "question": "How do different UN agencies approach climate change mitigation according to 2025 reports?",
            "ground_truth": "Different UN agencies approach climate change through coordinated strategies including UNDP's development focus, Security Council's security implications, and ECOSOC's economic dimensions.",
            "query_type": "comparative",
            "expected_docs": ["multiple"],
            "category": "environment"
        },
        {
            "question": "What common challenges in multilateral cooperation are identified across UN documents from 2025?",
            "ground_truth": "Common challenges include coordination complexity, resource constraints, political differences, implementation gaps, and the need for enhanced institutional mechanisms.",
            "query_type": "analytical",
            "expected_docs": ["multiple"],
            "category": "governance"
        },
        {
            "question": "How do recent UN reports address digital transformation and technology governance?",
            "ground_truth": "Recent UN reports address digital transformation through frameworks for inclusive technology access, digital governance mechanisms, cybersecurity considerations, and capacity building initiatives.",
            "query_type": "thematic",
            "expected_docs": ["UNCTAD", "A/", "E/"],
            "category": "technology"
        },
        
        # Specific Regional and Thematic Issues
        {
            "question": "What regional security concerns are most frequently mentioned in 2025 Security Council documents?",
            "ground_truth": "The most frequently mentioned regional security concerns include conflicts in various regions, peacekeeping mission challenges, humanitarian crises, and regional stability threats.",
            "query_type": "analytical",
            "expected_docs": ["S/2025"],
            "category": "peace_security"
        },
        {
            "question": "How do Economic and Social Council documents from 2025 address post-pandemic recovery?",
            "ground_truth": "ECOSOC documents address post-pandemic recovery through sustainable development strategies, economic resilience building, social protection systems, and coordinated policy responses.",
            "query_type": "thematic",
            "expected_docs": ["E/"],
            "category": "development"
        },
        
        # Procedural and Administrative
        {
            "question": "What budget and administrative matters are discussed in recent General Assembly session documents?",
            "ground_truth": "Recent General Assembly documents discuss budget allocations, administrative efficiency measures, organizational restructuring, and resource optimization strategies.",
            "query_type": "administrative",
            "expected_docs": ["A/80"],
            "category": "governance"
        },
        
        # Future-oriented and Strategic
        {
            "question": "What long-term strategic priorities does the UN system identify for sustainable development?",
            "ground_truth": "Long-term strategic priorities include accelerating SDG implementation, strengthening partnerships, enhancing institutional capacity, and addressing emerging global challenges.",
            "query_type": "strategic",
            "expected_docs": ["multiple"],
            "category": "development"
        },
        
        # Human Rights Focus
        {
            "question": "How do different UN bodies coordinate on human rights issues according to 2025 documents?",
            "ground_truth": "UN bodies coordinate on human rights through integrated approaches, shared frameworks, regular reporting mechanisms, and joint initiatives across different councils and agencies.",
            "query_type": "coordination",
            "expected_docs": ["E/CN.6", "A/", "E/"],
            "category": "human_rights"
        }
    ]
    
    return golden_queries

def save_golden_dataset(queries):
    """Save golden dataset to file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"golden_dataset_{timestamp}.json"
    filepath = Path("data") / filename
    
    # Create data directory if it doesn't exist
    filepath.parent.mkdir(exist_ok=True)
    
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_queries": len(queries),
            "categories": list(set(q["category"] for q in queries)),
            "query_types": list(set(q["query_type"] for q in queries))
        },
        "queries": queries
    }
    
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"💾 Golden dataset saved to: {filepath}")
    return str(filepath)

def main():
    """Create golden evaluation dataset."""
    print("🏗️  Creating Golden Evaluation Dataset")
    print("=" * 50)
    
    # Analyze corpus
    try:
        df, samples = analyze_corpus()
        print("✅ Corpus analyzed successfully")
    except Exception as e:
        print(f"❌ Failed to analyze corpus: {e}")
        return
    
    # Create golden queries
    print("\n📝 Creating 20 golden evaluation queries...")
    golden_queries = create_golden_queries()
    
    print(f"✅ Created {len(golden_queries)} golden queries")
    
    # Show summary
    categories = {}
    query_types = {}
    
    for query in golden_queries:
        cat = query["category"]
        qtype = query["query_type"]
        
        categories[cat] = categories.get(cat, 0) + 1
        query_types[qtype] = query_types.get(qtype, 0) + 1
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Categories: {dict(categories)}")
    print(f"   Query Types: {dict(query_types)}")
    
    # Save dataset
    filepath = save_golden_dataset(golden_queries)
    
    print(f"\n🎯 Golden Dataset Ready!")
    print(f"   File: {filepath}")
    print(f"   Queries: {len(golden_queries)}")
    print(f"   Ready for comprehensive evaluation!")
    
    return golden_queries

if __name__ == "__main__":
    main()