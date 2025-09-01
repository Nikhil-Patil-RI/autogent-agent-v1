import asyncio
import json
import aiohttp
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
from autogen_core.tools import FunctionTool
from typing import Annotated

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


async def generate_image_url(
    prompt: Annotated[str, "The text description of the image to generate"],
    model: Annotated[str, "Model name to use for generation"] = "flux",
    width: Annotated[int, "Width of the generated image"] = 1024,
    height: Annotated[int, "Height of the generated image"] = 1024,
    enhance: Annotated[bool, "Whether to enhance the prompt using an LLM"] = True,
    safe: Annotated[bool, "Whether to apply content filtering"] = False,
    seed: Annotated[int, "Seed for reproducible results"] = None
) -> str:
    """Generate an image URL from a text prompt using the Flux MCP server."""
    
    try:
        # Prepare the MCP request
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "generateImageUrl",
                "arguments": {
                    "prompt": prompt,
                    "model": model,
                    "width": width,
                    "height": height,
                    "enhance": enhance,
                    "safe": safe
                }
            }
        }
        
        # Add seed if provided
        if seed is not None:
            mcp_request["params"]["arguments"]["seed"] = seed
        
        # Make the HTTP request to the MCP server
        async with aiohttp.ClientSession() as session:
            async with session.post(
                IMAGE_URL,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result and "content" in result["result"]:
                        content = result["result"]["content"]
                        if isinstance(content, list) and len(content) > 0:
                            # Extract the image URL from the response
                            response_text = content[0].get("text", "")
                            try:
                                # Parse the JSON response to get the image URL
                                image_data = json.loads(response_text)
                                image_url = image_data.get("imageUrl", "")
                                if image_url:
                                    return f"✅ Image generated successfully!\n🖼️ Image URL: {image_url}\n📝 Prompt: {prompt}\n🎨 Model: {model}\n📐 Size: {width}x{height}"
                                else:
                                    return f"❌ Failed to generate image: No URL in response"
                            except json.JSONDecodeError:
                                return f"❌ Failed to parse image generation response: {response_text}"
                        else:
                            return f"❌ Failed to generate image: Empty response content"
                    else:
                        return f"❌ Failed to generate image: Invalid response format"
                else:
                    error_text = await response.text()
                    return f"❌ Failed to generate image: HTTP {response.status} - {error_text}"
                    
    except Exception as e:
        return f"❌ Error generating image: {str(e)}"


async def setup_tools():
    """Setup both search and custom image generation tools"""
    tools = []
    
    try:
        # Setup Search Tool (MCP)
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
        
        # Setup Custom Image Generation Tool (Direct HTTP)
        print("🔧 Setting up custom image generation tool...")
        image_tool = FunctionTool(generate_image_url, description="Generate an image URL from a text prompt")
        tools.append(image_tool)
        print("✅ Custom image generation tool created!")
        
        return tools
        
    except Exception as e:
        print(f"❌ Error setting up tools: {e}")
        return []


async def main() -> None:
    """Main function to run the multi-tool AutoGen agent"""
    
    # Setup tools
    tools = await setup_tools()
    
    if not tools:
        print("❌ Failed to setup tools. Exiting.")
        return
    
    # Create model client using Gemini
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    print("✅ Gemini model client created!")

    # Create multi-tool agent
    multi_agent = AssistantAgent(
        name="multi_tool_agent",
        model_client=model_client,
        tools=tools,
        system_message="""You are a helpful AI assistant with access to multiple tools:

1. **Search Tool**: Use this to search the web for information, current events, facts, news, etc.
   - Use when users ask questions about current events, facts, research, "what is", "who is", etc.
   - Examples: "What's the weather?", "Who won the election?", "Latest news about AI"

2. **Image Generation Tool (generate_image_url)**: Use this to create images based on text descriptions.
   - Use when users ask to create, generate, draw, or make images
   - Examples: "Create an image of a sunset", "Generate a picture of a cat", "Draw a futuristic city"
   - Required parameter: prompt (text description of the image)
   - Optional parameters: 
     * model: "flux" (default), "sdxl", "sd3", "sd15", "flux-schnell", "flux-dev"
     * width/height: numbers (default: 1024)
     * enhance: boolean (default: true) - enhances prompt using LLM
     * safe: boolean (default: false) - applies content filtering
     * seed: number for reproducible results

**Decision Making Guidelines:**
- If the user asks for information, facts, or current events → Use the search tool
- If the user asks to create, generate, or draw images → Use the generate_image_url tool
- If unclear, ask the user to clarify what they want
- Always be helpful and choose the most appropriate tool based on the user's request

When generating images, use descriptive prompts for better results and consider the optional parameters to customize the output.""",
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(name="user_proxy")
    
    # Create team
    team = RoundRobinGroupChat([multi_agent], max_turns=5)
    
    print(f"\n🤖 Multi-Tool AutoGen Agent (Gemini + Custom Tools) is ready! ({len(tools)} tools loaded)")
    print("I can help you with:")
    print("  🔍 Web search and information lookup")
    print("  🎨 Image generation from text descriptions")
    print("Type 'exit' to quit.\n")

    # Example queries
    print("💡 Example queries you can try:")
    print("  🔍 Search examples:")
    print("    • What's the latest news about AI?")
    print("    • Who won the 2024 Nobel Prize?")
    print("    • What is the current population of Tokyo?")
    print("  🎨 Image generation examples:")
    print("    • Create an image of a futuristic city")
    print("    • Generate a picture of a robotic cat working on computer")
    print("    • Draw a sunset over mountains")
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
