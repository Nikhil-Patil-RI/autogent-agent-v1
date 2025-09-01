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
from duckduckgo import base_url, params, url
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination

load_dotenv()


async def main() -> None:
    # Create server params for the remote MCP service
    server_params = StreamableHttpServerParams(
        url=url,
        headers={},
        timeout=30.0,  # HTTP timeout in seconds
        sse_read_timeout=300.0,  # SSE read timeout in seconds (5 minutes)
        terminate_on_close=True,
    )

    # Get the translation tool from the server
    adapter = await StreamableHttpMcpToolAdapter.from_server_params(
        server_params, "search"
    )

    # Create an agent that can use the translation tool
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    agent = AssistantAgent(
        name="search_agent",
        model_client=model_client,
        tools=[adapter],
        system_message="You are the helpful web scrapping agent. your task is to search query into web based on user query",
    )
    
    user_proxy = UserProxyAgent("user_proxy", input_func=input)
    
    termination = TextMentionTermination("APPROVE")
    
    
    team = RoundRobinGroupChat([agent], max_turns=1)

    task = "Write a 4-line poem about the ocean."
    while True:
        # Run the conversation and stream to the console.
        stream = team.run_stream(task=task)
        # Use asyncio.run(...) when running in a script.
        asyncio.run(Console(stream))
        # Get the user response.
        task = input("Enter your feedback (type 'exit' to leave): ")
        if task.lower().strip() == "exit":
            break
    # Let the agent translate some text
    # await Console(
    #     agent.run_stream(
    #         task="Translate 'Hello, how are you?' to Spanish",
    #         cancellation_token=CancellationToken(),
    #     )
    # )

if __name__ == "__main__":
    asyncio.run(main())
