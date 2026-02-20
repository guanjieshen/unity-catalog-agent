"""
Test script for Unity Catalog Data Advisor Agent.

This script provides examples of how to test the agent locally or in Databricks.
"""

from mlflow.types.responses import ResponsesAgentRequest

# Import the agent (this will initialize it)
from agent import AGENT


def test_basic_query():
    """Test a basic query to the agent."""
    print("=" * 60)
    print("Test 1: Basic Query - Peloton Data")
    print("=" * 60)
    
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What Peloton data do we have available? I need to analyze Peloton customer behavior and usage patterns."}],
        custom_inputs={"session_id": "test-session-1"}
    )
    
    response = AGENT.predict(request)
    print("\nResponse:")
    print(response.model_dump(exclude_none=True))
    print("\n")


def test_streaming_query():
    """Test a streaming query to the agent."""
    print("=" * 60)
    print("Test 2: Streaming Query")
    print("=" * 60)
    
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What tables contain sales data?"}],
        custom_inputs={"session_id": "test-session-2"}
    )
    
    print("\nStreaming response chunks:")
    for chunk in AGENT.predict_stream(request):
        chunk_data = chunk.model_dump(exclude_none=True)
        if chunk_data.get("type") == "response.output_item.done":
            item = chunk_data.get("item", {})
            if item.get("type") == "text":
                print(item.get("content", ""), end="", flush=True)
            elif item.get("type") == "function_call_output":
                print(f"\n[Tool Output]: {item.get('content', '')}")
    print("\n\n")


def test_data_discovery_query():
    """Test a data discovery query."""
    print("=" * 60)
    print("Test 3: Data Discovery Query")
    print("=" * 60)
    
    request = ResponsesAgentRequest(
        input=[{
            "role": "user", 
            "content": "I want to find datasets related to customer transactions and revenue"
        }],
        custom_inputs={"session_id": "test-session-3"}
    )
    
    response = AGENT.predict(request)
    
    # Extract and print the text content
    for output in response.output:
        if hasattr(output, 'content'):
            print(output.content)
        elif isinstance(output, dict) and 'content' in output:
            print(output['content'])
    print("\n")


def test_conversation():
    """Test a multi-turn conversation."""
    print("=" * 60)
    print("Test 4: Multi-turn Conversation")
    print("=" * 60)
    
    session_id = "test-conversation"
    
    # First message
    request1 = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What data do we have about products?"}],
        custom_inputs={"session_id": session_id}
    )
    
    response1 = AGENT.predict(request1)
    print("User: What data do we have about products?")
    print("Agent:", end=" ")
    for output in response1.output:
        if hasattr(output, 'content'):
            print(output.content)
        elif isinstance(output, dict) and 'content' in output:
            print(output['content'])
    print()
    
    # Follow-up message (in a real conversation, you'd include previous messages)
    request2 = ResponsesAgentRequest(
        input=[
            {"role": "user", "content": "What data do we have about products?"},
            {"role": "assistant", "content": "Based on the available data..."},  # Would be from response1
            {"role": "user", "content": "Can you show me the schema for the product table?"}
        ],
        custom_inputs={"session_id": session_id}
    )
    
    response2 = AGENT.predict(request2)
    print("User: Can you show me the schema for the product table?")
    print("Agent:", end=" ")
    for output in response2.output:
        if hasattr(output, 'content'):
            print(output.content)
        elif isinstance(output, dict) and 'content' in output:
            print(output['content'])
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Unity Catalog Data Advisor Agent - Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Run basic test with Peloton data query
        test_basic_query()
        
        # Uncomment other tests as needed
        # test_streaming_query()
        # test_data_discovery_query()
        # test_conversation()
        
        print("=" * 60)
        print("Tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure LLM_ENDPOINT_NAME is set correctly in config.py")
        print("2. Make sure VECTOR_SEARCH_INDEX_NAME is set correctly in config.py")
        print("3. Verify your Databricks credentials are correct")
        print("4. Check that you have access to the model serving endpoint")

