# Unity Catalog Data Advisor Agent

A Databricks agent that helps business users and analysts discover, understand, and safely use relevant datasets within Unity Catalog using vector search and foundation models.

## Overview

The Unity Catalog Data Advisor Agent uses semantic search to find relevant tables, schemas, and catalogs based on user queries. It provides business-friendly explanations and recommendations, helping users discover the right data without requiring deep technical knowledge.

## Features

- **Semantic Data Discovery**: Uses vector search to find relevant Unity Catalog datasets based on natural language queries
- **Business-Friendly Responses**: Explains datasets in non-technical terms, highlighting key metrics, dimensions, and use cases
- **Tool Integration**: Leverages Unity Catalog functions and vector search indexes as agent tools
- **MLflow Integration**: Deployable as an MLflow model with automatic authentication passthrough
- **Streaming Support**: Supports both streaming and non-streaming responses

## Architecture

The agent is built using the MLflow `ResponsesAgent` framework and consists of:

- **`config.py`**: Configuration for endpoints, system prompt, and tool settings
- **`tools.py`**: Tool setup including VectorSearchRetrieverTool and UC function tools
- **`agent.py`**: Main ToolCallingAgent class implementing ResponsesAgent interface
- **`log_model.py`**: Script to log the agent as an MLflow model with proper resources

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to a foundation model endpoint (e.g., Claude Opus)
- A vector search index containing Unity Catalog metadata
- Unity Catalog function `system.ai.python_exec` available

## Installation

### Option 1: Using Databricks Notebook (Recommended)

1. Upload all project files to your Databricks workspace (or clone the repo)
2. Open `test_agent_databricks.ipynb` in Databricks
3. Run the cells to test the agent

### Option 2: Manual Setup

1. Clone or download this repository to your Databricks workspace

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Before using the agent, configure the following in `config.py`:

### 1. Foundation Model Endpoint

Set your model serving endpoint name:

```python
LLM_ENDPOINT_NAME = "databricks-claude-opus-4-6"  # Replace with your endpoint
```

### 2. Vector Search Configuration

Set your vector search endpoint and index name:

```python
VECTOR_SEARCH_ENDPOINT_NAME = "data_advisor"  # Your vector search endpoint name
VECTOR_SEARCH_INDEX_NAME = "gshen_data_advisor.data_models.table_metadata"  # Your index name (format: catalog.schema.table)
```

The index name can be in the format `catalog.schema.table` or just the index name, depending on your setup.

### 3. Databricks Authentication (Optional)

**If running in a Databricks notebook or job**: Authentication is automatic. No configuration needed.

**If running locally or outside Databricks**: Set credentials via environment variables (recommended):

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"
```

Or create a `.env` file (make sure it's in `.gitignore`):
```
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token
```

To create a personal access token:
1. Go to User Settings → Access Tokens in Databricks
2. Click "Generate New Token"
3. Copy the token and set it as an environment variable

**Note**: Never commit tokens or credentials to version control. The config file uses environment variables by default for security.

### 4. System Prompt (Optional)

The system prompt is already configured for Unity Catalog data discovery, but you can customize it in `config.py` if needed.

## Usage

### Quick Test

Run the test script to verify everything is working:

```bash
python test_agent.py
```

### Local Testing

Test the agent locally before deployment:

```python
from agent import AGENT
from mlflow.types.responses import ResponsesAgentRequest

# Create a request
request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "I need to analyze customer purchase behavior"}],
    custom_inputs={"session_id": "test-session"}
)

# Get response
response = AGENT.predict(request)
print(response.model_dump(exclude_none=True))
```

### Streaming Responses

For real-time streaming responses:

```python
from agent import AGENT
from mlflow.types.responses import ResponsesAgentRequest

request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What tables contain sales data?"}],
    custom_inputs={"session_id": "stream-session"}
)

for chunk in AGENT.predict_stream(request):
    chunk_data = chunk.model_dump(exclude_none=True)
    if chunk_data.get("type") == "response.output_item.done":
        item = chunk_data.get("item", {})
        if item.get("type") == "text":
            print(item.get("content", ""), end="", flush=True)
```

### Testing in Databricks Notebook

If running in a Databricks notebook, you can test directly:

```python
# Cell 1: Import and test
from agent import AGENT
from mlflow.types.responses import ResponsesAgentRequest

# Cell 2: Simple test
result = AGENT.predict({
    "input": [{"role": "user", "content": "What datasets are available for sales analysis?"}],
    "custom_inputs": {"session_id": "notebook-test"}
})
print(result.model_dump(exclude_none=True))
```

## Deployment

### 1. Log the Agent as MLflow Model

```python
from log_model import log_agent_model

# Log the agent
logged_info = log_agent_model(
    model_name="unity-catalog-data-advisor",
    run_name="data-advisor-v1"
)

print(f"Model URI: {logged_info.model_uri}")
```

### 2. Register to Unity Catalog

```python
from log_model import register_model_to_uc

# Register the model
registered_info = register_model_to_uc(
    model_uri=logged_info.model_uri,
    catalog="main",
    schema="ml",
    model_name="data_advisor"
)

print(f"Registered model: {registered_info.name}")
print(f"Version: {registered_info.version}")
```

### 3. Deploy the Agent

```python
from databricks import agents

# Deploy the agent
agents.deploy(
    model_name="main.ml.data_advisor",
    model_version=registered_info.version,
    tags={"endpointSource": "data-advisor"},
    deploy_feedback_model=False,
)
```

### 4. Validate Before Deployment

Test the logged model before deploying:

```python
import mlflow

# Validate the model
result = mlflow.models.predict(
    model_uri=logged_info.model_uri,
    input_data={
        "input": [{"role": "user", "content": "Hello!"}],
        "custom_inputs": {"session_id": "validation-session"}
    },
    env_manager="uv",
)

print(result)
```

## Agent Behavior

The agent follows these guidelines:

1. **Understands user goals**: Clarifies business questions and identifies required metrics
2. **Recommends relevant datasets**: Searches across catalogs and schemas, preferring curated/production tables
3. **Explains in business terms**: Describes datasets, highlights key metrics, explains use cases
4. **Guides safe usage**: Calls out data grain, refresh frequency, limitations, and suggests joins
5. **Asks clarifying questions**: When objectives are ambiguous, asks focused follow-up questions

### Example Response Structure

```
User Objective: Analyze customer purchase behavior

Recommended Dataset(s):
- main.analytics.customer_purchases
- What this dataset represents: Daily aggregated customer purchase transactions
- Why it is relevant: Contains purchase amounts, product categories, and customer segments

Key Columns to Use:
- Primary metrics: purchase_amount, transaction_count
- Dimensions: customer_segment, product_category, region
- Date/time fields: purchase_date

Data Characteristics:
- Level of detail: Daily aggregates per customer
- Refresh cadence: Daily at 6 AM UTC
- Typical row volume: ~500K rows per day

How to Get Started:
- Example query: SELECT * FROM main.analytics.customer_purchases WHERE purchase_date >= CURRENT_DATE - 30
- Suggested filters: Filter by customer_segment for specific analysis
- Recommended joins: Join with main.analytics.customer_profiles on customer_id

Assumptions or Questions:
- Assumes daily granularity is sufficient (hourly data available in main.analytics.customer_purchases_hourly)
```

## Project Structure

```
data-advisor/
├── config.py           # Configuration and system prompt
├── tools.py         # Tool setup and factory functions
├── agent.py         # Main agent implementation
├── log_model.py     # MLflow logging utilities
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Troubleshooting

### Vector Search Not Working

- Verify `VECTOR_SEARCH_INDEX_NAME` is set correctly in `config.py`
- Ensure the vector search index is accessible from your workspace
- Check that the index contains Unity Catalog metadata

### Model Serving Endpoint Issues

- Verify `LLM_ENDPOINT_NAME` matches your endpoint name exactly
- Ensure you have permissions to access the endpoint
- Check endpoint status in Databricks UI

### Tool Execution Errors

- Verify `system.ai.python_exec` is available in your Unity Catalog
- Check Unity Catalog permissions for the function
- Review tool execution logs in MLflow traces

## Resources

- [Databricks Agent Framework Documentation](https://docs.databricks.com/en/generative-ai/agent-framework/author-agent.html)
- [MLflow ResponsesAgent API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)
- [Vector Search Integration](https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html)
- [Unity Catalog Functions](https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html)

## License

This project is provided as-is for use within Databricks workspaces.

