"""
Test script to debug AgentRunResponseUpdate structure
"""
import asyncio
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
import json

load_dotenv()

async def main():
    agent = AzureOpenAIChatClient(
        credential=AzureCliCredential()
    ).create_agent(
        instructions="You are a helpful assistant.",
        name="TestAgent"
    )
    
    print("Testing event structure...\n")
    
    async for event in agent.run_stream("Say hello in 3 words"):
        print(f"Event Type: {type(event).__name__}")
        print(f"Event Dir: {[attr for attr in dir(event) if not attr.startswith('_')]}")
        
        # Try to get text
        if hasattr(event, 'text'):
            print(f"event.text: {event.text}")
        if hasattr(event, 'delta'):
            print(f"event.delta: {event.delta}")
        if hasattr(event, 'content'):
            print(f"event.content: {event.content}")
            
        print(f"Full event: {event}")
        print("-" * 60)
        
        # Only show first 3 events
        break

if __name__ == "__main__":
    asyncio.run(main())
