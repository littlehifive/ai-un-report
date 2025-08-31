# 2025 Extended Corpus 

This folder is reserved for future expansion beyond the core 150 documents.

## Purpose
- **Current**: Core dataset has 150 high-impact 2025 UN reports
- **Future**: This folder will contain additional documents when expanded to 500-1000 reports

## Usage
The system currently uses only the `2025-core/` dataset. To enable extended corpus:

1. Update `config.yaml`:
```yaml
corpus:
  core_only: false  # Enable extended corpus
  target_documents: 500  # Increase target
```

2. Run discovery with extended parameters:
```bash
python src/discover_hybrid.py  # Will populate this folder
```

## Storage Strategy
- **Core (150 docs)**: Essential high-impact reports for demo
- **Extended (350+ docs)**: Comprehensive coverage for production use
- **Archives**: Historical documents from previous years

This approach keeps the demo manageable while allowing easy scaling to full production corpus.