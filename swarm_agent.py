"""
Agent Framework: Swarm Workflow as Agent
Demonstrates wrapping a HandoffBuilder workflow (swarm) as a single agent using workflow.as_agent()
Reference: https://learn.microsoft.com/en-us/agent-framework/user-guide/workflows/as-agents?pivots=programming-language-python

This takes the swarm workflow from 4_swarm.py and wraps it as a single agent.
"""

import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from agent_framework import AgentRunResponseUpdate, ai_function
from pydantic import Field
import logging

# Load environment variables
load_dotenv()


# Database Tools (from 4_swarm.py)
@ai_function
async def query_database(
    query: Annotated[str, Field(description="SQL query to execute")]
) -> str:
    """Execute a SQL query against the database."""
    if "sales" in query.lower():
        return "Query Results: Total Sales: $125,000 | Top Product: Widget Pro | Orders: 450"
    elif "customer" in query.lower():
        return "Query Results: Total Customers: 1,250 | Active: 980 | Churn Rate: 5.2%"
    elif "inventory" in query.lower():
        return "Query Results: Items in Stock: 3,450 | Low Stock Alerts: 12 | Warehouse: 85% capacity"
    else:
        return f"Query executed: {query} | Status: Success | Rows: 42"


@ai_function
async def get_database_schema(
    table_name: Annotated[str, Field(description="Name of the database table")] = None
) -> str:
    """Get database schema information for tables."""
    if table_name:
        return f"Schema for {table_name}: id (INT), name (VARCHAR), created_at (TIMESTAMP), status (VARCHAR)"
    return "Available tables: customers, orders, products, inventory, sales_analytics"


# Documentation Tools (from 4_swarm.py)
@ai_function
async def search_documentation(
    query: Annotated[str, Field(description="Search query for documentation")]
) -> str:
    """Search through documentation for relevant information."""
    query_lower = query.lower()
    if "api" in query_lower or "endpoint" in query_lower:
        return "API Documentation: Use /api/v1/customers for customer data. Authentication via Bearer token required."
    elif "setup" in query_lower or "install" in query_lower:
        return "Setup Guide: 1) Install dependencies 2) Configure .env file 3) Run database migrations 4) Start server"
    elif "authentication" in query_lower or "auth" in query_lower:
        return "Auth Documentation: System uses JWT tokens. Login at /auth/login with username/password. Token expires in 24h."
    else:
        return f"Documentation results for '{query}': Found 5 relevant articles covering implementation, best practices, and troubleshooting."


@ai_function
async def get_code_examples(
    topic: Annotated[str, Field(description="Topic to get code examples for")]
) -> str:
    """Retrieve code examples from documentation."""
    return f"""Code Example for {topic}:
```python
# Initialize connection
client = DatabaseClient(host='localhost')

# Execute query
result = client.query('SELECT * FROM {topic}')

# Process results
for row in result:
    print(row)
```"""
logging.basicConfig(filename='agent_events.log', level=logging.INFO)


def print_agent_event(event):
    """Extract text and metadata from agent events."""
    from agent_framework._types import TextContent, FunctionCallContent, FunctionResultContent
    
    text_delta = None
    function_call_info = None
    author_name = None
    content_type = None

    logging.info(f"Event type: {type(event)}, Event vars: {vars(event)}")
    
    # Handle AgentRunResponseUpdate streaming events
    if isinstance(event, AgentRunResponseUpdate):
        author_name = getattr(event, 'author_name', None)
        
        # Extract from contents list - check all content types
        contents = getattr(event, 'contents', None)
        function_results = []  # Collect all function results
        function_calls = []  # Collect all function calls
        
        if contents:
            for content in contents:
                if isinstance(content, TextContent):
                    text_delta = getattr(content, 'text', None)
                    if text_delta:
                        content_type = 'text'
                elif isinstance(content, FunctionCallContent):
                    # Extract function call information
                    func_name = getattr(content, 'name', None)
                    func_args = getattr(content, 'arguments', None)
                    if func_name:
                        function_calls.append({'name': func_name, 'arguments': func_args})
                        if not content_type:
                            content_type = 'function_call'
                elif isinstance(content, FunctionResultContent):
                    result = getattr(content, 'result', None)
                    call_id = getattr(content, 'call_id', None)
                    if result:
                        function_results.append({'result': result, 'call_id': call_id})
                        if not content_type:
                            content_type = 'function_result'
        
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
                        return {'text': text_delta, 'author': author_name, 'function_call': None, 'content_type': 'text'}
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


async def main():
    """Create a HandoffBuilder workflow and wrap it as an agent using .as_agent()"""
    from agent_framework import HandoffBuilder
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import AzureCliCredential
    
    # Clear/reset logging for fresh run
    log_file = "swarm_agent.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Configure logging to file only (no console output)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' mode clears file
        ],
        force=True  # Force reconfiguration if already configured
    )
    
    console = Console()
    
    print("=" * 60)
    print("Swarm Workflow as Agent Demo")
    print("=" * 60)
    
    # Create client
    client = AzureOpenAIChatClient(credential=AzureCliCredential())
    
    # Create specialist agents
    database_agent = client.create_agent(
        name="database_agent",
        instructions="You are a database specialist. Use your tools to query databases and retrieve schema information.",
        description="Handles database queries and data retrieval",
        tools=[query_database, get_database_schema],
    )
    
    document_agent = client.create_agent(
        name="document_agent",
        instructions="You are a documentation specialist. Use your tools to search documentation and provide code examples.",
        description="Handles documentation searches and code examples",
        tools=[search_documentation, get_code_examples],
    )
    
    # Create orchestrator
    orchestrator = client.create_agent(
        name="orchestrator",
        instructions=(
            "You are an orchestrator agent. Analyze the user's request and route to the appropriate specialist:\n"
            "- For database queries, data retrieval, or SQL questions: call handoff_to_database_agent\n"
            "- For documentation, setup guides, API references, or code examples: call handoff_to_document_agent\n"
        ),
        description="Routes requests to database or documentation specialists",
    )
    
    # Create HandoffBuilder workflow
    workflow = (
        HandoffBuilder(
            name="data_docs_handoff",
            participants=[orchestrator, database_agent, document_agent],
        )
        .set_coordinator(orchestrator)
        .add_handoff(orchestrator, [database_agent, document_agent])
        .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") > 100)
        .build()
    )
    
    # Wrap workflow as agent using .as_agent()
    print("\n✨ Converting workflow to agent using workflow.as_agent()...")
    workflow_agent = workflow.as_agent(name="SwarmWorkflowAgent")
    print(f"✅ Created workflow agent: {workflow_agent}")
    
    # Get a new thread for the workflow agent
    workflow_thread = workflow_agent.get_new_thread()
    print(f"✅ Created thread: {workflow_thread}\n")
    
    # Get initial user message
    rprint("[bold yellow]Enter your message (or 'quit' to exit):[/bold yellow]")
    user_input = input("> ").strip()
    
    if user_input.lower() == 'quit':
        print("Exiting...")
        return
    
    rprint("\n[bold yellow]User Message:[/bold yellow]")
    rprint(Panel(user_input, border_style="yellow", expand=False))
    
    accumulated_text = ""
    accumulated_function_calls = ""
    current_author = None
    previous_author = None
    
    with Live("", console=console, refresh_per_second=10) as live:
        live.update("")  # Start with empty/hidden display
        # Use run_stream on the workflow agent
        async for update in workflow_agent.run_stream(user_input, thread=workflow_thread):
            result = print_agent_event(update)
            if result:
                if result['author']:
                    # Check if author changed
                    if previous_author and result['author'] != previous_author:
                        # Finalize previous panel if there's accumulated text
                        if accumulated_text:
                            console.print(Panel(
                                accumulated_text,
                                title=f"[bold cyan]Response - {previous_author}[/bold cyan]",
                                border_style="cyan",
                                expand=False
                            ))
                            accumulated_text = ""
                            # Hide live panel while it's empty
                            live.update("")
                    
                    current_author = result['author']
                    previous_author = result['author']
                
                # Handle function calls separately
                if result.get('content_type') == 'function_call' and result.get('function_call'):
                    func_info = result['function_call']
                    func_name = func_info.get('name', 'unknown')
                    func_args = func_info.get('arguments', {})
                    
                    # Log function call panel to file
                    logging.info(f"[PANEL] Function Call - {current_author}: Calling {func_name} | Arguments: {func_args}")
                    
                    # Display function call in separate panel
                    console.print(Panel(
                        f"[bold magenta]Calling:[/bold magenta] {func_name}\n[dim]Arguments: {func_args}[/dim]",
                        title=f"[bold magenta]🔧 Function Call - {current_author}[/bold magenta]",
                        border_style="magenta",
                        expand=False
                    ))
                    accumulated_function_calls += f"{func_name}, "
                
                # Handle function results separately (may be multiple)
                elif result.get('content_type') == 'function_result' and result.get('function_results'):
                    # Display each function result in a separate panel
                    for func_result_info in result['function_results']:
                        func_result = func_result_info.get('result', 'N/A')
                        call_id = func_result_info.get('call_id', 'unknown')
                        
                        # Log function result panel to file
                        logging.info(f"[PANEL] Function Result - {current_author}: {func_result}")
                        
                        # Display function result in separate panel
                        console.print(Panel(
                            f"{func_result}",
                            title=f"[bold green]✓ Function Result - {current_author}[/bold green]",
                            border_style="green",
                            expand=False
                        ))
                
                # Handle text content
                elif result.get('text'):
                    accumulated_text += result['text']
                    
                    # Log text chunk to file
                    logging.info(f"[PANEL] Text Response - {current_author}: {result['text']}")
                    
                    # Update live panel with current author in title (only if there's text)
                    if accumulated_text:
                        panel_title = f"[bold cyan]Response - {current_author}[/bold cyan]" if current_author else "[bold cyan]Response[/bold cyan]"
                        live.update(Panel(
                            accumulated_text,
                            title=panel_title,
                            border_style="cyan",
                            expand=False
                        ))
    
    # Print final accumulated text if any remains
    if accumulated_text and current_author:
        console.print(Panel(
            accumulated_text,
            title=f"[bold cyan]Response - {current_author}[/bold cyan]",
            border_style="cyan",
            expand=False
        ))
    
    print()
    
    # Interactive loop - maintain conversational state (like 4_swarm.py)
    # Check if workflow agent has pending requests that need responses
    while workflow_agent.pending_requests:
        rprint("\n[bold yellow]Enter your message (or 'quit' to exit):[/bold yellow]")
        user_response = input("> ").strip()
        
        if user_response.lower() == 'quit':
            print("Exiting...")
            break
        
        rprint("\n[bold yellow]User Message:[/bold yellow]")
        rprint(Panel(user_response, border_style="yellow", expand=False))
        
        # Create function approval response messages for each pending request
        from agent_framework import ChatMessage, Role, FunctionApprovalResponseContent, FunctionCallContent
        
        response_messages = []
        for request_id, request_event in workflow_agent.pending_requests.items():
            # Get the original function call from the request
            function_call = FunctionCallContent(
                call_id=request_id,
                name=workflow_agent.REQUEST_INFO_FUNCTION_NAME,
                arguments={"request_id": request_id, "data": user_response}
            )
            
            # Create approval response
            approval_response = FunctionApprovalResponseContent(
                id=request_id,
                function_call=function_call,
                approved=True
            )
            
            response_messages.append(ChatMessage(
                role=Role.USER,
                contents=[approval_response]
            ))
        
        accumulated_text = ""
        accumulated_function_calls = ""
        current_author = None
        previous_author = None
        
        with Live("", console=console, refresh_per_second=10) as live:
            live.update("")  # Start with empty/hidden display
            # Continue with the same thread and send function approval responses
            async for update in workflow_agent.run_stream(response_messages, thread=workflow_thread):
                result = print_agent_event(update)
                if result:
                    if result['author']:
                        # Check if author changed
                        if previous_author and result['author'] != previous_author:
                            # Finalize previous panel if there's accumulated text
                            if accumulated_text:
                                console.print(Panel(
                                    accumulated_text,
                                    title=f"[bold cyan]Response - {previous_author}[/bold cyan]",
                                    border_style="cyan",
                                    expand=False
                                ))
                                accumulated_text = ""
                                # Hide live panel while it's empty
                                live.update("")
                        
                        current_author = result['author']
                        previous_author = result['author']
                    
                    # Handle function calls separately
                    if result.get('content_type') == 'function_call' and result.get('function_call'):
                        func_info = result['function_call']
                        func_name = func_info.get('name', 'unknown')
                        func_args = func_info.get('arguments', {})
                        
                        # Log function call panel to file
                        logging.info(f"[PANEL] Function Call - {current_author}: Calling {func_name} | Arguments: {func_args}")
                        
                        # Display function call in separate panel
                        console.print(Panel(
                            f"[bold magenta]Calling:[/bold magenta] {func_name}\n[dim]Arguments: {func_args}[/dim]",
                            title=f"[bold magenta]🔧 Function Call - {current_author}[/bold magenta]",
                            border_style="magenta",
                            expand=False
                        ))
                        accumulated_function_calls += f"{func_name}, "
                    
                    # Handle function results separately (may be multiple)
                    elif result.get('content_type') == 'function_result' and result.get('function_results'):
                        # Display each function result in a separate panel
                        for func_result_info in result['function_results']:
                            func_result = func_result_info.get('result', 'N/A')
                            call_id = func_result_info.get('call_id', 'unknown')
                            
                            # Log function result panel to file
                            logging.info(f"[PANEL] Function Result - {current_author}: {func_result}")
                            
                            # Display function result in separate panel
                            console.print(Panel(
                                f"{func_result}",
                                title=f"[bold green]✓ Function Result - {current_author}[/bold green]",
                                border_style="green",
                                expand=False
                            ))
                    
                    # Handle text content
                    elif result.get('text'):
                        accumulated_text += result['text']
                        
                        # Log text chunk to file
                        logging.info(f"[PANEL] Text Response - {current_author}: {result['text']}")
                        
                        # Update live panel with current author in title (only if there's text)
                        if accumulated_text:
                            panel_title = f"[bold cyan]Response - {current_author}[/bold cyan]" if current_author else "[bold cyan]Response[/bold cyan]"
                            live.update(Panel(
                                accumulated_text,
                                title=panel_title,
                                border_style="cyan",
                                expand=False
                            ))
        
        # Print final accumulated text if any remains
        if accumulated_text and current_author:
            console.print(Panel(
                accumulated_text,
                title=f"[bold cyan]Response - {current_author}[/bold cyan]",
                border_style="cyan",
                expand=False
            ))
        
        print()
    
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
