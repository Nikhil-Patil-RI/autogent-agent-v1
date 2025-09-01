import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StreamableHttpMcpToolAdapter,
    StreamableHttpServerParams,
)
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from urllib.parse import urlencode

# Load environment variables
load_dotenv()

# MCP Server Configuration (from duckduckgo.py)
BASE_URL = "https://server.smithery.ai/@nickclyde/duckduckgo-mcp-server/mcp"
MCP_PARAMS = {"api_key": "a9a64c1b-ddb1-4503-bb49-dfd7ab938c7b"}
MCP_URL = f"{BASE_URL}?{urlencode(MCP_PARAMS)}"


class AutoGenMCPAgent:
    """AutoGen Agent with MCP Tool Integration"""
    
    def __init__(self):
        self.model_client = None
        self.search_agent = None
        self.user_proxy = None
        self.team = None
        self.search_tool_adapter = None
    
    async def initialize(self):
        """Initialize the AutoGen agent with MCP tools"""
        try:
            # Create MCP server parameters
            server_params = StreamableHttpServerParams(
                url=MCP_URL,
                headers={},
                timeout=30.0,  # HTTP timeout in seconds
                sse_read_timeout=300.0,  # SSE read timeout in seconds (5 minutes)
                terminate_on_close=True,
            )
            
            # Get the search tool from the MCP server
            self.search_tool_adapter = await StreamableHttpMcpToolAdapter.from_server_params(
                server_params, "search"
            )
            
            # Create the model client (same as your original setup)
            self.model_client = OpenAIChatCompletionClient(
                model="gemini-1.5-flash-8b",
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            
            # Create the search agent with MCP tool
            self.search_agent = AssistantAgent(
                name="search_agent",
                model_client=self.model_client,
                tools=[self.search_tool_adapter],
                system_message="""You are a helpful web search agent. Your task is to search the web based on user queries using the DuckDuckGo search tool.

When a user asks a question:
1. Use the search tool to find relevant information
2. Analyze the search results
3. Provide a comprehensive and accurate answer based on the search results
4. Always cite your sources when possible

You have access to a search tool that can help you find current information on the web."""
            )
            
            # Create user proxy agent (simplified for AutoGen 0.7.4)
            self.user_proxy = UserProxyAgent(name="user_proxy")
            
            # Create team with the search agent
            self.team = RoundRobinGroupChat([self.search_agent], max_turns=3)
            
            print("✅ AutoGen MCP Agent initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing agent: {e}")
            return False
    
    async def search_and_respond(self, query: str):
        """Search for information and provide a response"""
        try:
            print(f"\n🔍 Processing query: {query}")
            
            # Run the conversation with the search agent
            stream = self.team.run_stream(task=query)
            await Console(stream)
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
    
    async def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n🤖 AutoGen MCP Search Agent Ready!")
        print("Type your questions and I'll search the web for answers.")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input
                user_query = input("You: ").strip()
                
                if user_query.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_query:
                    print("Please enter a valid query.")
                    continue
                
                # Process the query
                await self.search_and_respond(user_query)
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main function to run the AutoGen MCP Agent"""
    # Create and initialize the agent
    agent = AutoGenMCPAgent()
    
    # Initialize the agent with MCP tools
    if await agent.initialize():
        # Start interactive chat
        await agent.interactive_chat()
    else:
        print("Failed to initialize the agent. Please check your configuration.")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
