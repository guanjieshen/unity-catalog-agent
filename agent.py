"""
Unity Catalog Data Advisor Agent implementation.

This module contains the main ToolCallingAgent class that implements
the ResponsesAgent interface for discovering Unity Catalog datasets.
"""

import json
import warnings
from typing import Any, Generator
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    DATABRICKS_USERNAME,
    DATABRICKS_PASSWORD,
    LLM_ENDPOINT_NAME,
    SYSTEM_PROMPT,
)
from tools import ToolInfo, setup_tools


class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent for Unity Catalog data discovery.
    
    Handles both tool execution via exec_fn and LLM interactions via model serving.
    Uses vector search to find relevant tables, schemas, and catalogs.
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """
        Initializes the ToolCallingAgent with tools.
        
        Args:
            llm_endpoint: Name of the model serving endpoint
            tools: List of ToolInfo objects available to the agent
        """
        self.llm_endpoint = llm_endpoint
        self.workspace_client = self._get_workspace_client()
        self.model_serving_client: OpenAI = self._get_databricks_openai_client()
        self._tools_dict = {tool.name: tool for tool in tools}

    def _get_workspace_client(self) -> WorkspaceClient:
        """
        Get WorkspaceClient with appropriate authentication.
        
        Returns:
            WorkspaceClient instance
        """
        # If running in Databricks, WorkspaceClient() will auto-detect credentials
        # If running outside Databricks, use explicit credentials from config
        if DATABRICKS_HOST and DATABRICKS_TOKEN:
            return WorkspaceClient(
                host=DATABRICKS_HOST,
                token=DATABRICKS_TOKEN,
            )
        elif DATABRICKS_HOST and DATABRICKS_USERNAME and DATABRICKS_PASSWORD:
            return WorkspaceClient(
                host=DATABRICKS_HOST,
                username=DATABRICKS_USERNAME,
                password=DATABRICKS_PASSWORD,
            )
        else:
            # Auto-detect credentials (works in Databricks notebooks/jobs)
            return WorkspaceClient()

    def _get_databricks_openai_client(self) -> OpenAI:
        """
        Get OpenAI client configured for Databricks model serving.
        
        Returns:
            OpenAI client instance
        """
        # Initialize OpenAI client for Databricks model serving
        # Uses Databricks authentication from the workspace client
        host = self.workspace_client.config.host
        
        # Get token - try multiple methods for compatibility
        # In Databricks, the token might be in different places depending on auth method
        token = None
        
        # Method 1: Try config.token (works when explicitly set)
        if hasattr(self.workspace_client.config, 'token') and self.workspace_client.config.token:
            token = self.workspace_client.config.token
        
        # Method 2: Try api_client.token (works in some Databricks environments)
        elif hasattr(self.workspace_client, 'api_client') and hasattr(self.workspace_client.api_client, 'token'):
            token = self.workspace_client.api_client.token
        
        # Method 3: Try to get token from the HTTP session headers (common in Databricks)
        if not token and hasattr(self.workspace_client, 'api_client'):
            try:
                api_client = self.workspace_client.api_client
                # Try to get from session headers
                if hasattr(api_client, '_session') and api_client._session:
                    session = api_client._session
                    if hasattr(session, 'headers') and 'Authorization' in session.headers:
                        auth_header = session.headers.get('Authorization', '')
                        if auth_header.startswith('Bearer '):
                            token = auth_header.replace('Bearer ', '')
            except Exception:
                pass
        
        # Method 4: Try to get token from the auth provider
        if not token and hasattr(self.workspace_client.config, '_auth_provider'):
            try:
                auth_provider = self.workspace_client.config._auth_provider
                if hasattr(auth_provider, 'token'):
                    token = auth_provider.token
                elif hasattr(auth_provider, '_token'):
                    token = auth_provider._token
            except Exception:
                pass
        
        # Method 5: Fall back to DATABRICKS_TOKEN from config
        if not token and DATABRICKS_TOKEN:
            token = DATABRICKS_TOKEN
        
        # Method 6: Try environment variables
        if not token:
            try:
                import os
                token = os.getenv('DATABRICKS_TOKEN') or os.getenv('DATABRICKS_ACCESS_TOKEN')
            except Exception:
                pass
        
        # If still no token, raise a helpful error
        if not token:
            raise ValueError(
                "Databricks authentication token not found. "
                "In Databricks environments, ensure you're running in a notebook or job context. "
                "If running locally, set DATABRICKS_TOKEN environment variable. "
                f"Host: {host if host else 'Not found'}"
            )
        
        if not host:
            raise ValueError(
                "Databricks host not found. "
                "Please set DATABRICKS_HOST environment variable or ensure you're running in a Databricks environment."
            )
        
        # For Databricks model serving, use the serving-endpoints base URL
        # The model name will be passed to the chat.completions.create call
        base_url = f"{host}/serving-endpoints"
        
        return OpenAI(
            api_key=token,
            base_url=base_url,
        )

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        """
        Calls the LLM with the given messages and yields response chunks.
        
        Args:
            messages: List of message dictionaries in chat format
            
        Yields:
            Response chunks from the LLM
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                yield chunk.to_dict()

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output.
        
        Args:
            tool_call: Dictionary containing tool call information
            messages: Current message history
            
        Returns:
            ResponsesAgentStreamEvent with tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 15,  # Increased to allow more iterative searches
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Main tool calling loop that alternates between LLM calls and tool execution.
        
        This loop enables iterative refinement:
        1. LLM can call vector search to find initial datasets
        2. LLM analyzes search results and can call vector search again with refined queries
        3. Process continues until LLM has enough information to provide a comprehensive answer
        
        Args:
            messages: Current message history (includes all tool results)
            max_iter: Maximum number of iterations before stopping (increased to 15 for better iteration)
            
        Yields:
            ResponsesAgentStreamEvent objects
        """
        iteration_count = 0
        for _ in range(max_iter):
            iteration_count += 1
            last_msg = messages[-1]
            
            # If the last message is from assistant (final answer), we're done
            if last_msg.get("role", None) == "assistant":
                return
            
            # If the last message is a function call, execute it
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
                # After tool execution, continue the loop to let LLM process the results
                # The LLM will see the tool output and can decide to call more tools or provide an answer
            
            # Otherwise, call the LLM (it may decide to call tools or provide an answer)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Predict method for non-streaming requests.
        
        Args:
            request: ResponsesAgentRequest containing input messages
            
        Returns:
            ResponsesAgentResponse with output items
        """
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Predict method for streaming requests.
        
        Args:
            request: ResponsesAgentRequest containing input messages
            
        Yields:
            ResponsesAgentStreamEvent objects
        """
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]
        yield from self.call_and_run_tools(messages=messages)


def create_agent():
    """
    Factory function to create and configure the agent.
    
    Returns:
        ToolCallingAgent instance
    """
    tools = setup_tools()
    return ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=tools)


# Initialize agent when module is imported
# IMPORTANT: Do NOT call mlflow.openai.autolog() here at module import time.
# It causes initialization errors because MLflow's _multi_processor is not initialized yet.
# Instead, call mlflow.openai.autolog() in the notebook/script that uses the agent
# (see test_agent_databricks.ipynb) or in log_model.py when logging the model.
AGENT = create_agent()

