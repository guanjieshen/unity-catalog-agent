"""
Tool setup for Unity Catalog Data Advisor Agent.

This module handles:
- ToolInfo class definition
- Factory function for creating tools
- Unity Catalog function toolkit integration
- Vector search retriever tool setup
"""

from typing import Any, Callable, Optional

from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

from config import UC_TOOL_NAMES, VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_ENDPOINT_NAME


class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None, uc_function_client=None):
    """
    Factory function to create ToolInfo objects from a given tool spec
    and (optionally) a custom execution function.
    
    Args:
        tool_spec: Tool specification dictionary
        exec_fn_param: Optional custom execution function
        uc_function_client: Unity Catalog function client (required if exec_fn_param is None)
    """
    # Remove 'strict' property, as Claude models do not support it in tool specs.
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    # Converts tool name with double underscores to UDF dot notation.
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        # Return error message if execution fails, result value if not.
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value

    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


# Global list to store vector search tool instances for resource declaration
VECTOR_SEARCH_TOOLS = []


def setup_tools():
    """
    Set up all tools for the agent including UC functions and vector search.
    
    Returns:
        list[ToolInfo]: List of all configured tools
    """
    global VECTOR_SEARCH_TOOLS
    tool_infos = []
    
    # Set up Unity Catalog function tools
    uc_function_client = get_uc_function_client()
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    for tool_spec in uc_toolkit.tools:
        tool_infos.append(create_tool_info(tool_spec, uc_function_client=uc_function_client))
    
    # Set up vector search retriever tools
    # This enables semantic search across Unity Catalog metadata
    VECTOR_SEARCH_TOOLS.clear()
    
    if VECTOR_SEARCH_INDEX_NAME:
        # VectorSearchRetrieverTool supports endpoint_name and index_name
        # The index_name can be in format: catalog.schema.table or just the index name
        vs_tool = VectorSearchRetrieverTool(
            index_name=VECTOR_SEARCH_INDEX_NAME,
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME if VECTOR_SEARCH_ENDPOINT_NAME else None,
            # Add filters here if needed, e.g., filters="catalog='main'"
        )
        VECTOR_SEARCH_TOOLS.append(vs_tool)
        tool_infos.append(create_tool_info(vs_tool.tool, vs_tool.execute))
    
    return tool_infos


# Export the tool setup function and ToolInfo class
__all__ = ["ToolInfo", "create_tool_info", "setup_tools", "VECTOR_SEARCH_TOOLS"]

