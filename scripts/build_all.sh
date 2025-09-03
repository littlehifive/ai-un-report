#!/bin/bash

# UN Reports RAG - Full Pipeline Build Script
# This script runs the complete pipeline: discover -> fetch -> parse -> index

set -e  # Exit on any error

echo "🇺🇳 Starting UN Reports RAG Pipeline..."
echo "======================================"

# Check if we're in the right directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found. Please run from project root directory."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Step 1: Discovery
echo ""
echo "🔍 Step 1: Discovering UN reports..."
echo "-----------------------------------"
cd src
python discover.py
if [ $? -ne 0 ]; then
    echo "❌ Discovery failed!"
    exit 1
fi
cd ..

# Step 2: Fetch
echo ""
echo "📥 Step 2: Downloading report files..."
echo "-------------------------------------"
cd src
python fetch.py
if [ $? -ne 0 ]; then
    echo "❌ Fetch failed!"
    exit 1
fi
cd ..

# Step 3: Parse
echo ""
echo "📄 Step 3: Parsing documents..."
echo "------------------------------"
cd src
python parse.py
if [ $? -ne 0 ]; then
    echo "❌ Parsing failed!"
    exit 1
fi
cd ..

# Step 4: Index
echo ""
echo "🔍 Step 4: Creating embeddings index..."
echo "--------------------------------------"
cd src
python indexer.py
if [ $? -ne 0 ]; then
    echo "❌ Indexing failed!"
    exit 1
fi
cd ..

echo ""
echo "✅ Pipeline completed successfully!"
echo ""
echo "📊 Summary:"
echo "----------"

# Show basic stats if files exist
if [ -f "data/records.parquet" ]; then
    echo "📋 Records file: data/records.parquet"
fi

if [ -f "data/raw/files_manifest.parquet" ]; then
    echo "📥 Files manifest: data/raw/files_manifest.parquet"
fi

if [ -f "data/parsed/chunks.parquet" ]; then
    echo "📄 Parsed chunks: data/parsed/chunks.parquet"
fi

if [ -f "data/index.faiss" ]; then
    echo "🔍 FAISS index: data/index.faiss"
fi

if [ -f "data/index.meta.json" ]; then
    echo "📋 Index metadata: data/index.meta.json"
fi

echo ""
echo "🚀 Ready to run: streamlit run src/app.py"
echo ""