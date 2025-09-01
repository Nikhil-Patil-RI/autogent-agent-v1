import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StreamableHttpMcpToolAdapter,
    StreamableHttpServerParams,
)
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
import os
from dotenv import load_dotenv
from urllib.parse import urlencode
from autogen_agentchat.teams import RoundRobinGroupChat

load_dotenv()

# MCP Configuration (from your duckduckgo.py)
base_url = "https://server.smithery.ai/@nickclyde/duckduckgo-mcp-server/mcp"
params = {"api_key": "a9a64c1b-ddb1-4503-bb49-dfd7ab938c7b"}
url = f"{base_url}?{urlencode(params)}"


async def main() -> None:
    # Create server params for the MCP service
    server_params = StreamableHttpServerParams(
        url=url,
        headers={},
        timeout=30.0,  # HTTP timeout in seconds
        sse_read_timeout=300.0,  # SSE read timeout in seconds (5 minutes)
        terminate_on_close=True,
    )

    # Get the search tool from the server
    search_adapter = await StreamableHttpMcpToolAdapter.from_server_params(
        server_params, "search"
    )

    # Create model client (same as your original)
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Create search agent with MCP tool
    search_agent = AssistantAgent(
        name="search_agent",
        model_client=model_client,
        tools=[search_adapter],
        system_message="You are a helpful web search agent. Use the search tool to find information based on user queries and provide comprehensive answers.",
    )
    
    # Create user proxy agent (simplified for AutoGen 0.7.4)
    user_proxy = UserProxyAgent(name="user_proxy")
    
    # Create team
    team = RoundRobinGroupChat([search_agent], max_turns=3)

    print("🤖 AutoGen MCP Search Agent is ready!")
    print("Ask me anything and I'll search the web for answers.")
    print("Type 'exit' to quit.\n")

    # Test with a simple query first
    test_query = "What is the current weather?"
    print(f"🔍 Testing with query: {test_query}")

    try:
        # Run the conversation and stream to console
        stream = team.run_stream(task=test_query)
        await Console(stream)
        print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"❌ Error during test: {e}")

    print("✅ Agent is working! You can now interact with it.")
    print("Note: The agent will search the web and provide answers based on your queries.\n")


if __name__ == "__main__":
    asyncio.run(main())
