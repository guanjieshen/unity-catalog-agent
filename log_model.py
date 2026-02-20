"""
MLflow model logging utilities for Unity Catalog Data Advisor Agent.

This module provides functions to log the agent as an MLflow model with
proper resource declarations for automatic authentication passthrough.
"""

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksVectorSearchIndex
from pkg_resources import get_distribution

from config import UC_TOOL_NAMES, VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_ENDPOINT_NAME
from tools import VECTOR_SEARCH_TOOLS


def get_model_resources():
    """
    Determine Databricks resources to specify for automatic auth passthrough at deployment time.
    
    Returns:
        list: List of resource objects for MLflow model logging
    """
    resources = []
    
    # Add vector search index resources
    for tool in VECTOR_SEARCH_TOOLS:
        resources.extend(tool.resources)
    
    # If VECTOR_SEARCH_TOOLS is empty but index name is configured, add it directly
    if not resources and VECTOR_SEARCH_INDEX_NAME:
        resources.append(DatabricksVectorSearchIndex(index_name=VECTOR_SEARCH_INDEX_NAME))
    
    # Add Unity Catalog function resources
    for tool_name in UC_TOOL_NAMES:
        resources.append(DatabricksFunction(function_name=tool_name))
    
    return resources


def log_agent_model(model_name: str = "unity-catalog-data-advisor", run_name: str = None):
    """
    Log the agent as an MLflow model with proper resource declarations.
    
    Args:
        model_name: Name for the logged model
        run_name: Optional name for the MLflow run
        
    Returns:
        ModelInfo object containing information about the logged model
    """
    # Enable MLflow OpenAI autologging before logging the model
    import mlflow
    mlflow.openai.autolog()
    
    # Ensure tools are set up to populate VECTOR_SEARCH_TOOLS
    from tools import setup_tools
    setup_tools()
    
    resources = get_model_resources()
    
    # Get databricks-connect version for pip requirements
    try:
        databricks_connect_version = get_distribution("databricks-connect").version
    except Exception:
        # Fallback if databricks-connect is not installed
        databricks_connect_version = None
    
    pip_requirements = [
        "databricks-openai",
        "backoff",
        "mlflow-skinny[databricks]",
        "databricks-agents",
    ]
    
    if databricks_connect_version:
        pip_requirements.append(f"databricks-connect=={databricks_connect_version}")
    else:
        pip_requirements.append("databricks-connect")
    
    with mlflow.start_run(run_name=run_name):
        logged_agent_info = mlflow.pyfunc.log_model(
            name=model_name,
            python_model="agent.py",
            pip_requirements=pip_requirements,
            resources=resources,
        )
    
    return logged_agent_info


def register_model_to_uc(
    model_uri: str,
    catalog: str,
    schema: str,
    model_name: str,
):
    """
    Register the logged model to Unity Catalog.
    
    Args:
        model_uri: URI of the logged model (e.g., from log_agent_model)
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        model_name: Name for the registered model
        
    Returns:
        ModelVersion object containing information about the registered model
    """
    mlflow.set_registry_uri("databricks-uc")
    
    uc_model_name = f"{catalog}.{schema}.{model_name}"
    
    registered_model_info = mlflow.register_model(
        model_uri=model_uri,
        name=uc_model_name
    )
    
    return registered_model_info


if __name__ == "__main__":
    # Example usage
    print("Logging Unity Catalog Data Advisor Agent as MLflow model...")
    logged_info = log_agent_model()
    print(f"Model logged successfully!")
    print(f"Model URI: {logged_info.model_uri}")
    print(f"Run ID: {logged_info.run_id}")
    print("\nTo register to Unity Catalog, use:")
    print("  from log_model import register_model_to_uc")
    print(f"  register_model_to_uc('{logged_info.model_uri}', 'your_catalog', 'your_schema', 'your_model_name')")

