"""
Agent as Tool Pattern Example
Demonstrates converting agents into tools that can be used by other agents
"""
import asyncio
import os
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.console import Console
from agent_framework import ChatAgent, AgentRunResponseUpdate, ChatResponseUpdate
from agent_framework.azure import AzureOpenAIChatClient, AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
import json
# Load environment variables
load_dotenv()


def print_agent_event(event):
    """Extract text and metadata from MAF agent events."""
    event_type_str = type(event).__name__
    print(vars(event))
    
    # Print detailed content structure from contents if available
    contents = getattr(event, 'contents', None)
    if contents:
        from agent_framework._types import TextContent, FunctionCallContent, UsageContent, FunctionResultContent
        print(f"\n=== Contents List (Total: {len(contents)}) ===")
        for idx, content in enumerate(contents):
            print(f"\n[Content {idx}] Type: {type(content).__name__}")
            
            if isinstance(content, TextContent):
                print(f"  TextContent structure:")
                print(f"    text: {getattr(content, 'text', None)}")
                print(f"    All attributes: {vars(content)}")
                
            elif isinstance(content, FunctionCallContent):
                print(f"  FunctionCallContent structure:")
                print(f"    name: {getattr(content, 'name', None)}")
                print(f"    call_id: {getattr(content, 'call_id', None)}")
                print(f"    arguments: {getattr(content, 'arguments', None)}")
                print(f"    All attributes: {vars(content)}")
                
            elif isinstance(content, UsageContent):
                print(f"  UsageContent structure:")
                print(f"    input_tokens: {getattr(content, 'input_tokens', None)}")
                print(f"    output_tokens: {getattr(content, 'output_tokens', None)}")
                print(f"    total_tokens: {getattr(content, 'total_tokens', None)}")
                print(f"    All attributes: {vars(content)}")
                
            elif isinstance(content, FunctionResultContent):
                print(f"  FunctionResultContent structure:")
                print(f"    call_id: {getattr(content, 'call_id', None)}")
                print(f"    result: {getattr(content, 'result', None)}")
                print(f"    All attributes: {vars(content)}")
            else:
                print(f"  Unknown content type")
                print(f"    All attributes: {vars(content)}")
        print("=" * 50)
    
    print("\n")

    # Handle AgentRunResponseUpdate streaming events
    if isinstance(event, AgentRunResponseUpdate):
        # Get text content
        text_delta = getattr(event, 'text', None)
        
        # Get author name
        author_name = getattr(event, 'author_name', None)
        
        # Get role
        role = getattr(event, 'role', None)
        
        # Check if this is a function call (tool call)
        contents = getattr(event, 'contents', None)
        if contents and not text_delta:
            # This might be a function call - extract function name
            from agent_framework._types import FunctionCallContent
            for content in contents:
                if isinstance(content, FunctionCallContent):
                    func_name = getattr(content, 'name', 'unknown_function')
                    # Return a special marker for function calls
                    return {
                        'text': f"[Calling {func_name}...]",
                        'author': author_name,
                        'role': role,
                        'event_type': event_type_str,
                        'is_function_call': True
                    }
        
        return {
            'text': text_delta,
            'author': author_name,
            'role': role,
            'event_type': event_type_str,
            'is_function_call': False
        }
    
    # For other event types, try to extract text
    if hasattr(event, 'message'):
        msg = event.message
        if msg and hasattr(msg, 'text') and msg.text:
            return {'text': msg.text, 'author': None, 'role': None, 'event_type': event_type_str, 'is_function_call': False}
    elif hasattr(event, 'text') and event.text:
        return {'text': event.text, 'author': None, 'role': None, 'event_type': event_type_str, 'is_function_call': False}
    
    return None


async def main():
    console = Console()
    
    print("=" * 60)
    print("Agent as Tool Pattern Demo")
    print("=" * 60)
    
    # Check which deployment to use (same logic as file 1)
    responses_deployment = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    
    # Create the appropriate client based on available deployment
    if responses_deployment:
        print(f"Using AzureOpenAIResponsesClient with deployment: {responses_deployment}")
        chat_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())
    elif chat_deployment:
        print(f"Using AzureOpenAIChatClient with deployment: {chat_deployment}")
        chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    else:
        print("Using AzureOpenAIChatClient with default deployment")
        chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    
    # Create specialized agents
    print("\n--- Creating Specialized Agents ---")
    
    # 1. Creative Writer Agent
    writer = ChatAgent(
        name="CreativeWriter",
        chat_client=chat_client,
        instructions=(
            "You are a creative writer specializing in engaging, imaginative content. "
            "Write in a vivid, descriptive style that captures the reader's attention."
        )
    )
    
    # 2. Technical Expert Agent
    tech_expert = ChatAgent(
        name="TechExpert",
        chat_client=chat_client,
        instructions=(
            "You are a technical expert with deep knowledge of programming, systems, and technology. "
            "Provide accurate, detailed technical explanations and recommendations."
        )
    )
    
    # 3. Editor Agent
    editor = ChatAgent(
        name="Editor",
        chat_client=chat_client,
        instructions=(
            "You are a professional editor. Review content for clarity, grammar, and structure. "
            "Provide constructive feedback and polished versions of text."
        )
    )
    
    # Convert agents to tools
    print("--- Converting Agents to Tools ---")
    
    writer_tool = writer.as_tool(
        name="creative_writer",
        description="Generate creative, engaging content on any topic",
        arg_name="request",
        arg_description="What to write about"
    )
    
    tech_expert_tool = tech_expert.as_tool(
        name="technical_expert",
        description="Get technical expertise and recommendations on programming and technology topics",
        arg_name="question",
        arg_description="Technical question to answer"
    )
    
    editor_tool = editor.as_tool(
        name="editor",
        description="Review and polish written content for clarity and quality",
        arg_name="content",
        arg_description="Content to review and improve"
    )
    
    # Create coordinator agent that uses other agents as tools
    print("--- Creating Coordinator Agent ---")
    
    coordinator = ChatAgent(
        name="Coordinator",
        chat_client=chat_client,
        instructions=(
            "You are a project coordinator managing a team of specialists. "
            "You have access to a creative writer, technical expert, and editor. "
            "Delegate tasks to the appropriate specialist and synthesize their work. "
            "Always use the editor to polish final outputs."
        ),
        tools=[writer_tool, tech_expert_tool, editor_tool]
    )
    
    # Helper function to process agent streaming responses
    async def process_agent_stream(agent, task, example_title):
        rprint("\n" + "=" * 60)
        rprint(f"[bold green]{example_title}[/bold green]")
        rprint("=" * 60)
        
        rprint(f"\n[bold yellow]Task:[/bold yellow]")
        rprint(Panel(task, border_style="yellow", expand=False))
        
        accumulated_text = ""
        current_author = None
        live_context = None
        
        async for event in agent.run_stream(task):
            result = print_agent_event(event)
            if result and result['text']:
                # Check if author changed
                if result['author'] and result['author'] != current_author:
                    # Stop current live panel if exists
                    if live_context:
                        live_context.stop()
                        # Print the completed panel
                        rprint(Panel(
                            accumulated_text,
                            title=f"[bold cyan]{current_author} Response[/bold cyan]",
                            subtitle=f"[dim]Author: {current_author}[/dim]",
                            border_style="cyan",
                            expand=False
                        ))
                        print()
                    
                    # Start new panel for new author
                    current_author = result['author']
                    accumulated_text = result['text']
                    live_context = Live(Panel("", title=f"[bold cyan]{current_author} Response[/bold cyan]", border_style="cyan"), console=console, refresh_per_second=10)
                    live_context.start()
                else:
                    # Continue accumulating text for current author
                    accumulated_text += result['text']
                    if result['author']:
                        current_author = result['author']
                
                # Update the live panel
                if live_context:
                    subtitle = f"[dim]Author: {current_author}[/dim]" if current_author else None
                    live_context.update(Panel(
                        accumulated_text,
                        title=f"[bold cyan]{current_author or 'Agent'} Response[/bold cyan]",
                        subtitle=subtitle,
                        border_style="cyan",
                        expand=False
                    ))
        
        # Print final panel
        if live_context:
            live_context.stop()
            if accumulated_text:
                rprint(Panel(
                    accumulated_text,
                    title=f"[bold cyan]{current_author or 'Agent'} Response[/bold cyan]",
                    subtitle=f"[dim]Author: {current_author}[/dim]" if current_author else None,
                    border_style="cyan",
                    expand=False
                ))
        
        print()
    
    # Define tasks
    tasks = [
        {
            "title": "Example 1: Creative Writing Task",
            "task": "write a 1 line joke on ai trends",
            "agent": coordinator
        },
        # {
        #     "title": "Example 2: Technical Question",
        #     "task": "Ask the technical expert about the differences between async/await and threading in Python.",
        #     "agent": coordinator
        # },
        # {
        #     "title": "Example 3: Multi-Agent Workflow",
        #     "task": (
        #         "Create a short blog post introduction about microservices architecture. "
        #         "First, get technical details from the expert, then have the writer create "
        #         "engaging content, and finally have the editor polish it."
        #     ),
        #     "agent": coordinator
        # },
        # {
        #     "title": "Example 4: Direct Writer Call",
        #     "task": "Write one sentence about quantum computing",
        #     "agent": writer
        # },
        # {
        #     "title": "Example 5: Coordinator Delegating to Writer",
        #     "task": "Use the creative writer to write one sentence about quantum computing",
        #     "agent": coordinator
        # }
    ]
    
    # Process all tasks
    for item in tasks:
        await process_agent_stream(item["agent"], item["task"], item["title"])
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    rprint("\n[bold cyan]Key Takeaways:[/bold cyan]")
    rprint("✓ Agents can be converted to tools using .as_tool()")
    rprint("✓ Tool-based agents can be used by coordinator agents")
    rprint("✓ This enables hierarchical agent architectures")
    rprint("✓ Coordinator agents can orchestrate complex multi-step workflows")


if __name__ == "__main__":
    asyncio.run(main())
