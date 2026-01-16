
import os
import sys
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.googlesearch import GoogleSearchTools

# Check for API keys in env
print("Checking Environment Variables:")
print(f"SERPAPI_API_KEY present: {'SERPAPI_API_KEY' in os.environ}")
print(f"GOOGLE_API_KEY present: {'GOOGLE_API_KEY' in os.environ}")
print("-" * 50)

try:
    print("Initializing Agent with GoogleSearchTools...")
    agent = Agent(
        model=Ollama(id="mistral"),
        tools=[GoogleSearchTools()],
        show_tool_calls=True,  # Changed to True to see what happens
        markdown=True
    )
    
    query = "yolo object detection optimization stm32"
    print(f"Running query: '{query}'")
    
    response = agent.run(f"Search for: {query}\nReturn top 3 URLs.")
    
    print("-" * 50)
    print("RAW RESPONSE:")
    print(response)
    print("-" * 50)
    
    if hasattr(response, 'content'):
        print("RESPONSE CONTENT:")
        print(response.content)
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
