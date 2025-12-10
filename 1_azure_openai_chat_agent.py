"""
Azure OpenAI Chat Completion Agent Example
Demonstrates basic chat agent with function tools and streaming
"""
import asyncio
import os
from typing import Annotated
from agent_framework.azure import AzureOpenAIChatClient, AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from pydantic import Field
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from agent_framework import AgentRunResponseUpdate, ai_function

# Load environment variables
load_dotenv()


def print_agent_event(event):
    """Extract text and metadata from agent events."""
    from agent_framework._types import TextContent, FunctionCallContent, FunctionResultContent, UsageContent
    
    text_delta = None
    function_call_info = None
    author_name = None
    content_type = None
    
    # Handle AgentRunResponseUpdate streaming events
    if isinstance(event, AgentRunResponseUpdate):
        author_name = getattr(event, 'author_name', None)
        
        # Extract from contents list - check all content types
        contents = getattr(event, 'contents', None)
        function_results = []  # Collect all function results
        function_calls = []  # Collect all function calls
        usage_detected = False
        
        if contents:
            for content in contents:
                if isinstance(content, TextContent):
                    text_delta = getattr(content, 'text', None)
                    if text_delta:
                        content_type = 'text'
                elif isinstance(content, FunctionCallContent):
                    # Extract function call information
                    func_name = getattr(content, 'name', None) or ''
                    func_args = getattr(content, 'arguments', None) or ''
                    call_id = getattr(content, 'call_id', None) or ''
                    # Always append even if name is empty (for streaming chunks)
                    function_calls.append({'name': func_name, 'arguments': func_args, 'call_id': call_id})
                    if not content_type:
                        content_type = 'function_call'
                elif isinstance(content, FunctionResultContent):
                    result = getattr(content, 'result', None)
                    call_id = getattr(content, 'call_id', None)
                    if result:
                        function_results.append({'result': result, 'call_id': call_id})
                        if not content_type:
                            content_type = 'function_result'
                elif isinstance(content, UsageContent):
                    # UsageContent signals completion of LLM stream
                    usage_detected = True
                    content_type = 'usage'
        
        # Return usage signal if detected
        if usage_detected:
            return {
                'text': None,
                'author': author_name,
                'function_call': None,
                'function_calls': None,
                'function_result': None,
                'function_results': None,
                'content_type': 'usage'
            }
        
        # Return first function call if any
        if function_calls:
            return {
                'text': None,
                'author': author_name,
                'function_call': function_calls[0],
                'function_calls': function_calls,
                'function_result': None,
                'function_results': None,
                'content_type': 'function_call'
            }
        
        # Return all function results if any
        if function_results:
            return {
                'text': None,
                'author': author_name,
                'function_call': None,
                'function_calls': None,
                'function_result': function_results[0],
                'function_results': function_results,
                'content_type': 'function_result'
            }
        
        return {
            'text': text_delta, 
            'author': author_name, 
            'function_call': function_call_info,
            'function_calls': None,
            'function_result': None,
            'function_results': None,
            'content_type': content_type
        }
    
    # Handle nested event.data structures
    if hasattr(event, 'data') and event.data:
        data = event.data
        if isinstance(data, AgentRunResponseUpdate):
            return print_agent_event(data)
        
        author_name = getattr(data, 'author_name', None)
        if hasattr(data, 'contents') and data.contents:
            for content in data.contents:
                if isinstance(content, TextContent):
                    text_delta = getattr(content, 'text', None)
                    if text_delta:
                        return {'text': text_delta, 'author': author_name, 'function_call': None, 'function_result': None, 'content_type': 'text'}
                elif isinstance(content, FunctionCallContent):
                    func_name = getattr(content, 'name', None)
                    func_args = getattr(content, 'arguments', None)
                    if func_name:
                        return {
                            'text': None, 
                            'author': author_name, 
                            'function_call': {'name': func_name, 'arguments': func_args},
                            'function_result': None,
                            'content_type': 'function_call'
                        }
                elif isinstance(content, FunctionResultContent):
                    result = getattr(content, 'result', None)
                    call_id = getattr(content, 'call_id', None)
                    if result:
                        return {
                            'text': None,
                            'author': author_name,
                            'function_call': None,
                            'function_result': {'result': result, 'call_id': call_id},
                            'content_type': 'function_result'
                        }
    
    return None


@ai_function
async def get_weather(
    location: Annotated[str, Field(description="The location to get weather for")]
) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: sunny, 25°C"


@ai_function
async def calculate_tip(
    bill_amount: Annotated[float, Field(description="The total bill amount")],
    tip_percentage: Annotated[float, Field(description="Tip percentage (e.g., 15 for 15%)")] = 15.0
) -> str:
    """Calculate tip amount for a bill."""
    print("tool call made")
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"


async def main():
    from rich.live import Live
    from rich.console import Console
    
    console = Console()
    
    print("=" * 60)
    print("Azure OpenAI Chat Completion Agent Demo")
    print("=" * 60)
    
    # Check which deployment to use
    responses_deployment = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    
    # Create the appropriate client based on available deployment
    if responses_deployment:
        print(f"Using AzureOpenAIResponsesClient with deployment: {responses_deployment}")
        client = AzureOpenAIResponsesClient(credential=AzureCliCredential())
    elif chat_deployment:
        print(f"Using AzureOpenAIChatClient with deployment: {chat_deployment}")
        client = AzureOpenAIChatClient(credential=AzureCliCredential())
    else:
        print("Using AzureOpenAIChatClient with default deployment")
        client = AzureOpenAIChatClient(credential=AzureCliCredential())
    
    # Create agent with function tools
    agent = client.create_agent(
        instructions="You are a helpful assistant that can check weather and calculate tips.",
        name="HelpfulAssistant",
        tools=[get_weather, calculate_tip]
    )
    
    # Get a new thread for conversation
    thread = agent.get_new_thread()
    
    # Interactive conversation loop
    rprint("\n[bold cyan]Interactive Chat (type 'quit' to exit)[/bold cyan]\n")
    
    while True:
        # Get user input
        rprint("\n[bold yellow]Enter your message (or 'quit' to exit):[/bold yellow]")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        # Display user message in panel
        rprint("\n[bold yellow]User Message:[/bold yellow]")
        rprint(Panel(user_input, border_style="yellow", expand=False))
        rprint()  # Add spacing
        
        accumulated_text = ""
        current_author = None
        
        # Function call accumulation per call_id
        function_call_accumulator = {}  # {call_id: {'name': str, 'arguments': str}}
        last_call_id = None  # Track the last valid call_id for chunks with empty call_id
        
        with Live("", console=console, refresh_per_second=10) as live:
            live.update("")  # Start with empty/hidden display
            async for event in agent.run_stream(user_input, thread=thread):
                result = print_agent_event(event)
                if result:
                    if result['author']:
                        current_author = result['author']
                    
                    # Handle function calls - accumulate arguments across chunks
                    if result.get('content_type') == 'function_call' and result.get('function_call'):
                        func_info = result['function_call']
                        func_name = func_info.get('name', '')
                        func_args = func_info.get('arguments', '')
                        call_id = func_info.get('call_id', '')
                        
                        # Use last_call_id if current call_id is empty (subsequent chunks)
                        if not call_id and last_call_id:
                            call_id = last_call_id
                        elif call_id:
                            last_call_id = call_id
                        else:
                            call_id = 'default'
                        
                        # Initialize or update accumulator for this call_id
                        if call_id not in function_call_accumulator:
                            function_call_accumulator[call_id] = {'name': '', 'arguments': '', 'author': current_author}
                        
                        if func_name:
                            function_call_accumulator[call_id]['name'] = func_name
                        if func_args:
                            function_call_accumulator[call_id]['arguments'] += str(func_args)
                    
                    # Handle usage content - signals LLM stream completion, display accumulated function calls
                    elif result.get('content_type') == 'usage':
                        # Display any accumulated function calls now that streaming is complete
                        for call_id, call_data in function_call_accumulator.items():
                            if call_data['name']:  # Only display if we have a name
                                func_name = call_data['name']
                                func_args = call_data['arguments']
                                call_author = call_data.get('author', current_author)
                                
                                # Display function call in separate panel
                                console.print(Panel(
                                    f"[bold magenta]Calling:[/bold magenta] {func_name}\n[dim]Arguments: {func_args}[/dim]",
                                    title=f"[bold magenta]🔧 Function Call - {call_author}[/bold magenta]",
                                    border_style="magenta",
                                    expand=False
                                ))
                        
                        # Clear accumulator and reset call_id tracking after displaying
                        function_call_accumulator.clear()
                        last_call_id = None
                    
                    # Handle function results
                    elif result.get('content_type') == 'function_result' and result.get('function_results'):
                        # Display function results (may be multiple)
                        for func_result_info in result['function_results']:
                            func_result = func_result_info.get('result', 'N/A')
                            call_id = func_result_info.get('call_id', 'unknown')
                            
                            # Display function result in separate panel
                            console.print(Panel(
                                f"{func_result}",
                                title=f"[bold green]✓ Function Result - {current_author}[/bold green]",
                                border_style="green",
                                expand=False
                            ))
                    
                    # Handle text content
                    elif result.get('text'):
                        # Don't display function calls here - they will be displayed when we get usage content
                        # Just accumulate the text
                        accumulated_text += result['text']
                        
                        # Update live panel with current author in title (only if there's text)
                        if accumulated_text:
                            panel_title = f"[bold cyan]Response - {current_author}[/bold cyan]" if current_author else "[bold cyan]Response[/bold cyan]"
                            live.update(Panel(
                                accumulated_text,
                                title=panel_title,
                                border_style="cyan",
                                expand=False
                            ))
        
        print()  # Add spacing between turns
    
    print("=" * 60)
    print("Chat ended!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
