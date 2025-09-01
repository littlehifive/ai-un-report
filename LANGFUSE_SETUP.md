# Langfuse Integration Setup

## 1. Install Dependencies
```bash
pip install ragas langfuse datasets
```

## 2. Langfuse Cloud Setup

Since you already have a Langfuse account:

1. **Go to your Langfuse dashboard**: https://cloud.langfuse.com
2. **Create/Select Project**: Create a new project for "UN Reports RAG"
3. **Get API Keys**:
   - Go to Settings → API Keys
   - Copy your `Public Key` and `Secret Key`

## 3. Environment Variables Setup

Add to your `.env` file:
```bash
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY="pk_lf_..."
LANGFUSE_SECRET_KEY="sk_lf_..."
LANGFUSE_HOST="https://cloud.langfuse.com"

# Your existing OpenAI key
OPENAI_API_KEY="sk-..."
```

## 4. Integration Features

### Automatic Tracking
- **Conversations**: Each chat session tracked with unique ID
- **Search Operations**: Query → retrieval → ranking metrics
- **Generation**: Model calls with token usage and parameters
- **Errors**: Failed operations and debugging info

### Manual Features
- **User Feedback**: Thumbs up/down on responses
- **A/B Testing**: Compare different prompts or models
- **Performance Monitoring**: Response times, costs, quality scores

## 5. Running Evaluations

### RAGAS Evaluation
```bash
cd src
python eval_ragas.py
```

This will:
- Run 7 test queries covering different scenarios
- Calculate RAGAS metrics (faithfulness, relevancy, etc.)
- Save detailed results to `data/ragas_evaluation_YYYYMMDD_HHMMSS.json`

### Expected RAGAS Metrics
- **Faithfulness**: 0.7-0.9 (responses grounded in context)
- **Answer Relevancy**: 0.6-0.8 (answers match questions)  
- **Context Precision**: 0.5-0.8 (retrieved docs are relevant)
- **Context Recall**: 0.4-0.7 (all relevant docs retrieved)

### Langfuse Dashboard
After running conversations, check your Langfuse dashboard for:
- **Traces**: Individual conversation flows
- **Generations**: Model API calls with costs
- **Scores**: RAGAS metrics automatically uploaded
- **Sessions**: User conversation patterns

## 6. Adding to Streamlit App

To integrate with your main app, add this to `app.py`:

```python
from langfuse_integration import langfuse_tracker

# In your chat logic:
trace_id = langfuse_tracker.start_conversation(
    session_id=st.session_state.get("session_id", "default"),
    user_id="demo_user"
)

# Track search
langfuse_tracker.track_search(trace_id, query, results, {"top_k": top_k})

# Track generation  
langfuse_tracker.track_generation(trace_id, query, results, response, {
    "model": "gpt-4o-mini",
    "temperature": 0.1
})
```

## 7. Monitoring Production

Once deployed, you'll see:
- **Real user conversations** in Langfuse
- **Cost tracking** per conversation
- **Quality metrics** over time
- **Error rates** and debugging info

This gives you complete observability into your RAG system performance!