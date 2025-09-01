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

# MCP Server Configurations
# DuckDuckGo Search Tool
SEARCH_BASE_URL = "https://server.smithery.ai/@nickclyde/duckduckgo-mcp-server/mcp"
SEARCH_PARAMS = {"api_key": "a9a64c1b-ddb1-4503-bb49-dfd7ab938c7b"}
SEARCH_URL = f"{SEARCH_BASE_URL}?{urlencode(SEARCH_PARAMS)}"

# Flux Image Generation Tool
IMAGE_BASE_URL = "https://server.smithery.ai/@falahgs/flux-imagegen-mcp-server/mcp"
IMAGE_PARAMS = {"api_key": "a9a64c1b-ddb1-4503-bb49-dfd7ab938c7b"}
IMAGE_URL = f"{IMAGE_BASE_URL}?{urlencode(IMAGE_PARAMS)}"


async def setup_mcp_tools():
    """Setup MCP tools: search (and image generation if compatible)"""
    tools = []

    try:
        # Setup Search Tool
        print("🔧 Setting up DuckDuckGo search tool...")
        search_server_params = StreamableHttpServerParams(
            url=SEARCH_URL,
            headers={},
            timeout=30.0,
            sse_read_timeout=300.0,
            terminate_on_close=True,
        )

        search_adapter = await StreamableHttpMcpToolAdapter.from_server_params(
            search_server_params, "search"
        )
        tools.append(search_adapter)
        print("✅ Search tool connected!")

        # Setup Image Generation Tool with proper schema
        print("🔧 Setting up Flux image generation tool...")
        try:
            image_server_params = StreamableHttpServerParams(
                url=IMAGE_URL,
                headers={},
                timeout=30.0,
                sse_read_timeout=300.0,
                terminate_on_close=True,
            )

            image_adapter = await StreamableHttpMcpToolAdapter.from_server_params(
                image_server_params, "generateImageUrl"
            )
            tools.append(image_adapter)
            print("✅ Image generation tool connected!")

        except Exception as e:
            print(f"⚠️ Image generation tool failed: {e}")
            print("   Continuing with search tool only...")

        return tools

    except Exception as e:
        print(f"❌ Error setting up MCP tools: {e}")
        return []


async def main() -> None:
    """Main function to run the multi-tool AutoGen agent"""
    
    # Setup MCP tools
    tools = await setup_mcp_tools()
    
    if not tools:
        print("❌ Failed to setup MCP tools. Exiting.")
        return
    
    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    print("✅ Model client created!")

    # Determine available tools for system message
    has_search = any("search" in str(tool) for tool in tools)
    has_image = any("generate" in str(tool).lower() for tool in tools)

    # Create dynamic system message based on available tools
    system_message = "You are a helpful AI assistant with access to the following tools:\n\n"

    if has_search:
        system_message += """1. **Search Tool**: Use this to search the web for information, current events, facts, news, etc.
   - Use when users ask questions about current events, facts, research, "what is", "who is", etc.
   - Examples: "What's the weather?", "Who won the election?", "Latest news about AI"\n\n"""

    if has_image:
        system_message += """2. **Image Generation Tool (generateImageUrl)**: Use this to create images based on text descriptions.
   - Use when users ask to create, generate, draw, or make images
   - Examples: "Create an image of a sunset", "Generate a picture of a cat", "Draw a futuristic city"
   - Required parameter: prompt (text description of the image)
   - Optional parameters:
     * model: "flux" (default), "sdxl", "sd3", "sd15", "flux-schnell", "flux-dev"
     * width/height: numbers (default: 1024)
     * enhance: boolean (default: true) - enhances prompt using LLM
     * safe: boolean (default: false) - applies content filtering
     * seed: number for reproducible results\n\n"""

    system_message += """**Decision Making Guidelines:**"""
    if has_search:
        system_message += "\n- If the user asks for information, facts, or current events → Use the search tool"
    if has_image:
        system_message += "\n- If the user asks to create, generate, or draw images → Use the image generation tool"
    if not has_image:
        system_message += "\n- For image generation requests, explain that image generation is currently unavailable due to technical limitations, but offer to search for information about the topic instead"

    system_message += "\n- If unclear, ask the user to clarify what they want\n- Always be helpful and choose the most appropriate tool based on the user's request"

    # Create multi-tool agent
    multi_agent = AssistantAgent(
        name="multi_tool_agent",
        model_client=model_client,
        tools=tools,
        system_message=system_message,
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(name="user_proxy")
    
    # Create team
    team = RoundRobinGroupChat([multi_agent], max_turns=5)

    print(f"\n🤖 Multi-Tool AutoGen Agent (Gemini) is ready! ({len(tools)} tools loaded)")
    print("I can help you with:")
    if has_search:
        print("  🔍 Web search and information lookup")
    if has_image:
        print("  🎨 Image generation from text descriptions")
    if not has_image:
        print("  ⚠️  Image generation currently unavailable (schema compatibility issue)")
    print("Type 'exit' to quit.\n")

    # Show example queries based on available tools
    print("💡 Example queries you can try:")
    if has_search:
        print("  🔍 Search examples:")
        print("    • What's the latest news about AI?")
        print("    • Who won the 2024 Nobel Prize?")
        print("    • What is the current population of Tokyo?")
    if has_image:
        print("  🎨 Image generation examples:")
        print("    • Create an image of a futuristic city")
        print("    • Generate a picture of a sunset over mountains")
        print("    • Draw a cute robot playing with a cat")
    elif has_search:
        print("  📝 For images, I can search for information about:")
        print("    • Image generation tools and techniques")
        print("    • Art styles and visual concepts")
        print("    • Photography and design inspiration")
    print()

    # Interactive loop
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
            
            print(f"\n🤖 Processing: {user_query}")
            print("-" * 50)
            
            # Run the conversation
            stream = team.run_stream(task=user_query)
            await Console(stream)
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    asyncio.run(main())
