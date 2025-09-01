# Run RAGAS Evaluation with Langfuse Dashboard

## Step-by-Step Guide

### Step 1: Add Your API Keys
Edit the `.env` file in your project root and replace the placeholder keys:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_actual_openai_key_here

# Langfuse Configuration (replace with your actual keys)
LANGFUSE_PUBLIC_KEY="pk_lf_your_actual_public_key_here"
LANGFUSE_SECRET_KEY="sk_lf_your_actual_secret_key_here"
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### Step 2: Test Langfuse Connection
```bash
python test_langfuse_connection.py
```

You should see:
```
✅ API keys found
✅ Test trace created
✅ Test generation created
✅ Langfuse tracker is enabled
🎉 SUCCESS! Langfuse is working correctly.
```

### Step 3: Run RAGAS Evaluation with Langfuse Tracking
```bash
python src/eval_ragas_langfuse.py
```

This will:
- ✅ Run 7 comprehensive test queries
- ✅ Calculate RAGAS metrics (faithfulness, relevancy, etc.)
- ✅ Send all data to your Langfuse dashboard
- ✅ Save detailed results locally

Expected output:
```
🧮 Running RAGAS evaluation (this may take a few minutes)...
🎯 RAGAS EVALUATION RESULTS WITH LANGFUSE TRACKING
================================================================
📊 Total Queries: 7
✅ Successful Retrievals: 5-7
📄 Avg Documents Retrieved: 4-6

🧮 RAGAS METRICS:
🟢 faithfulness: 0.750+ - Factual accuracy
🟢 answer_relevancy: 0.650+ - Answer relevance  
🟡 context_precision: 0.600+ - Context precision
...
```

### Step 4: View Results on Langfuse Dashboard

1. **Go to your Langfuse dashboard**: https://cloud.langfuse.com
2. **Select your project** (e.g., "UN Reports RAG")
3. **Look for traces** with session ID like `ragas_eval_20250901_123456`

**What you'll see:**
- 🔍 **Individual traces** for each evaluation query
- 📊 **RAGAS metrics** as scores on each trace
- ⏱️ **Response times** and token usage
- 🎯 **Search performance** with similarity scores
- 📝 **Full conversation context** (question → search → generation)

### Step 5: Analyze Your Results

**In Langfuse Dashboard:**
- **Traces tab**: See individual query executions
- **Sessions tab**: View the full evaluation session
- **Scores tab**: RAGAS metrics breakdown
- **Analytics**: Performance trends over time

**Key Metrics to Watch:**
- 🎯 **Faithfulness > 0.7**: Low hallucination risk
- 🎯 **Answer Relevancy > 0.6**: Good question matching
- 🎯 **Context Precision > 0.5**: Relevant document retrieval

### Troubleshooting

**If connection test fails:**
1. Double-check your API keys in `.env`
2. Ensure no extra spaces around the keys
3. Verify you're using the correct Langfuse project

**If RAGAS evaluation is slow:**
- It processes 7 complex queries with OpenAI API calls
- Expected time: 3-5 minutes
- Each query involves: embedding → search → GPT-4o-mini generation

**If you see low scores:**
- Faithfulness < 0.6: Check for hallucinations
- Relevancy < 0.5: Questions may not match your corpus  
- Precision < 0.4: Consider adjusting search thresholds

### Next Steps

Once evaluation completes successfully:
1. ✅ **View results** in Langfuse dashboard
2. ✅ **Analyze RAGAS metrics** for system performance
3. ✅ **Set up continuous monitoring** for production use
4. ✅ **A/B test** different models or prompts

Your UN Reports RAG system is now fully monitored and evaluated! 🎉