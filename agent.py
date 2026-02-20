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
        # WorkspaceClient() auto-detects credentials in Databricks environments
        # For local testing, set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
        self.workspace_client = WorkspaceClient()
        # Use workspace client's built-in method to get OpenAI client with authentication
        # This automatically handles authentication in Databricks environments
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            # Convert messages to chat completions format
            # This handles the conversion from Responses format to OpenAI format
            chat_messages = to_chat_completions_input(messages)
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=chat_messages,
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        try:
            args = json.loads(tool_call.get("arguments"))
        except Exception:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 15,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                # Handle ALL pending tool calls, not just the last one.
                # The LLM may emit multiple parallel function_call items at once,
                # and ALL must have corresponding results before the next LLM call.
                pending_tool_calls = []
                for msg in reversed(messages):
                    if msg.get("type") == "function_call":
                        pending_tool_calls.append(msg)
                    else:
                        break
                pending_tool_calls.reverse()
                for tool_call in pending_tool_calls:
                    yield self.handle_tool_call(tool_call, messages)
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

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
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

