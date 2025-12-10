"""
Agent Framework HandoffBuilder Pattern
Demonstrates agent handoff coordination with specialized agents:
- Orchestrator: Routes queries to appropriate specialists
- Database Agent: Queries database using tools
- Document Agent: Searches documentation using tools
"""

import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.console import Console
from agent_framework import AgentRunResponseUpdate, ai_function
from pydantic import Field

# Load environment variables
load_dotenv()


# Database Tools
@ai_function
async def query_database(
    query: Annotated[str, Field(description="SQL query to execute")]
) -> str:
    """Execute a SQL query against the database."""
    # Simulate database query
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


# Documentation Tools
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


async def main() -> None:
    """Agent Framework's HandoffBuilder for agent coordination."""
    from agent_framework import (
        AgentRunUpdateEvent,
        HandoffBuilder,
        HandoffUserInputRequest,
        RequestInfoEvent,
        WorkflowRunState,
        WorkflowStatusEvent,
    )
    from agent_framework.azure import AzureOpenAIChatClient, AzureOpenAIResponsesClient
    from azure.identity import AzureCliCredential, ClientSecretCredential, DefaultAzureCredential
    
    
    console = Console()
    
    print("=" * 60)
    print("Agent Framework Handoff Pattern Demo")
    print("=" * 60)

    # Check which deployment to use (same logic as file 1)
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

    # Create Database Agent with database tools
    database_agent = client.create_agent(
        name="database_agent",
        instructions=(
            "You are a database specialist. Use your tools to query databases and retrieve schema information. "
            "Execute SQL queries and provide data analysis. Always use the query_database tool for data retrieval."
        ),
        description="Handles database queries and data retrieval",
        tools=[query_database, get_database_schema],
    )

    # Create Document Agent with documentation tools
    document_agent = client.create_agent(
        name="document_agent",
        instructions=(
            "You are a documentation specialist. Use your tools to search documentation and provide code examples. "
            "Help users find information in docs, guides, and API references. Always use your search tools."
        ),
        description="Handles documentation searches and code examples",
        tools=[search_documentation, get_code_examples],
    )

    # Create Data Analysis Agent using existing Azure AI Foundry agent
    # This assumes you have an existing agent in Azure AI Foundry
    # Replace <your-agent-id> with your actual agent ID from Azure AI Foundry
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureAIAgentClient
    from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential
    
    print("\n--- Creating Data Analysis Agent from Azure AI Foundry ---")
    data_analysis_agent_id = os.getenv("AGENT_ID")
    
    if data_analysis_agent_id == os.getenv("AGENT_ID"):
        print("⚠️  Warning: AZURE_AI_FOUNDRY_AGENT_ID not set in .env")
        print("⚠️  Set it to use an existing Azure AI Foundry agent")
        print("⚠️  Creating a placeholder agent instead...")
        
        # Fallback to regular agent if no Azure AI Foundry agent ID provided
        data_analysis_agent = client.create_agent(
            name="data_analysis_agent",
            instructions=(
                "You are a data analysis specialist. Analyze data patterns, generate insights, "
                "create visualizations, and provide statistical analysis. Help with data interpretation."
            ),
            description="Handles data analysis and insights generation",
        )
    else:
        print(f"Using Azure AI Foundry agent ID: {data_analysis_agent_id}")
        # Use existing Azure AI Foundry agent
        credential = ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        )
        credential=DefaultAzureCredential()
        data_analysis_agent = ChatAgent(
            chat_client=AzureAIAgentClient(
                name="data_analysis_agent",
                async_credential=credential,
                agent_id=data_analysis_agent_id
            ),
            instructions="You are a data analysis specialist from Azure AI Foundry."
        )

    # Create Orchestrator agent
    orchestrator = client.create_agent(
        name="orchestrator",
        instructions=(
            "You are an orchestrator agent. Analyze the user's request and route to the appropriate specialist:\n"
            "- For database queries, data retrieval, or SQL questions: call handoff_to_database_agent\n"
            "- For documentation, setup guides, API references, or code examples: call handoff_to_document_agent\n"
            "- For data analysis, insights, patterns, statistics, or visualizations: call handoff_to_data_analysis_agent\n"
            "Analyze the request carefully and delegate to the most appropriate specialist."
        ),
        description="Routes requests to database, documentation, or data analysis specialists",
    )

    # Create handoff workflow
    # Orchestrator routes to database_agent, document_agent, or data_analysis_agent
    workflow = (
        HandoffBuilder(
            name="data_docs_analysis_handoff",
            participants=[orchestrator, database_agent, document_agent, data_analysis_agent],
        )
        .set_coordinator(orchestrator)
        .add_handoff(orchestrator, [database_agent, document_agent, data_analysis_agent])
        .add_handoff(database_agent, orchestrator)
        .add_handoff(document_agent, orchestrator)
        .add_handoff(data_analysis_agent, orchestrator)
        .with_termination_condition(lambda conv: sum(1 for msg in conv if msg.role.value == "user") > 100)
        .enable_return_to_previous(True)
        .build()
    )


    # Get initial user message from terminal
    rprint("\n[bold yellow]Enter your message (or 'quit' to exit):[/bold yellow]")
    user_input = input("> ").strip()
    
    if user_input.lower() == 'quit':
        print("Exiting...")
        return
    
    rprint("\n[bold yellow]User Message:[/bold yellow]")
    rprint(Panel(user_input, border_style="yellow", expand=False))

    current_executor = None
    accumulated_text = ""
    author_name = None
    pending_requests: list[RequestInfoEvent] = []

    with Live(Panel("", title="[bold cyan]Agent Response[/bold cyan]", border_style="cyan"), console=console, refresh_per_second=10) as live:
        async for event in workflow.run_stream(user_input):
            if isinstance(event, AgentRunUpdateEvent):
                # Check if executor changed
                if current_executor != event.executor_id:
                    if accumulated_text:  # Print previous agent's response
                        live.stop()
                        rprint(Panel(
                            accumulated_text,
                            title=f"[bold cyan]{current_executor}[/bold cyan]",
                            subtitle=f"[dim]Author: {author_name}[/dim]" if author_name else None,
                            border_style="cyan",
                            expand=False
                        ))
                    
                    # Reset for new agent
                    current_executor = event.executor_id
                    accumulated_text = ""
                    author_name = None
                
                # Extract text from event.data (AgentRunResponseUpdate)
                if event.data:
                    result = print_agent_event(event.data)
                    if result:
                        if result['text']:
                            accumulated_text += result['text']
                        if result['author']:
                            author_name = result['author']
                    
                    # Update live panel
                    subtitle = None
                    if author_name:
                        subtitle = f"[dim]Author: {author_name}[/dim]"
                    
                    live.update(Panel(
                        accumulated_text,
                        title=f"[bold cyan]{current_executor}[/bold cyan]",
                        subtitle=subtitle,
                        border_style="cyan",
                        expand=False
                    ))
            elif isinstance(event, RequestInfoEvent):
                if isinstance(event.data, HandoffUserInputRequest):
                    pending_requests.append(event)
    
    # Print the final agent's panel after the stream ends
    # if accumulated_text:
    #     rprint(Panel(
    #         accumulated_text,
    #         title=f"[bold cyan]{current_executor}[/bold cyan]",
    #         subtitle=f"[dim]Author: {author_name}[/dim]" if author_name else None,
    #         border_style="cyan",
    #         expand=False
    #     ))
    
    print()

    # Interactive loop - keep asking for user input
    while pending_requests:
        rprint("\n[bold yellow]Enter your message (or 'quit' to exit):[/bold yellow]")
        user_response = input("> ").strip()
        
        if user_response.lower() == 'quit':
            print("Exiting...")
            break
        
        rprint("\n[bold yellow]User Message:[/bold yellow]")
        rprint(Panel(user_response, border_style="yellow", expand=False))

        responses = {req.request_id: user_response for req in pending_requests}
        pending_requests = []
        current_executor = None
        accumulated_text = ""
        author_name = None

        with Live(Panel("", title="[bold cyan]Agent Response[/bold cyan]", border_style="cyan"), console=console, refresh_per_second=10) as live:
            async for event in workflow.send_responses_streaming(responses):
                if isinstance(event, AgentRunUpdateEvent):
                    # Check if executor changed
                    if current_executor != event.executor_id:
                        if accumulated_text:  # Print previous agent's response
                            live.stop()
                            rprint(Panel(
                                accumulated_text,
                                title=f"[bold cyan]{current_executor}[/bold cyan]",
                                subtitle=f"[dim]Author: {author_name}[/dim]" if author_name else None,
                                border_style="cyan",
                                expand=False
                            ))
                        
                        # Reset for new agent
                        current_executor = event.executor_id
                        accumulated_text = ""
                        author_name = None
                    
                    # Extract text from event.data (AgentRunResponseUpdate)
                    if event.data:
                        result = print_agent_event(event.data)
                        if result:
                            if result['text']:
                                accumulated_text += result['text']
                            if result['author']:
                                author_name = result['author']
                        
                        # Update live panel
                        subtitle = None
                        if author_name:
                            subtitle = f"[dim]Author: {author_name}[/dim]"
                        
                        live.update(Panel(
                            accumulated_text,
                            title=f"[bold cyan]{current_executor}[/bold cyan]",
                            subtitle=subtitle,
                            border_style="cyan",
                            expand=False
                        ))
                elif isinstance(event, RequestInfoEvent):
                    if isinstance(event.data, HandoffUserInputRequest):
                        pending_requests.append(event)
        
        # Print the final agent's panel after the stream ends
        # if accumulated_text:
        #     rprint(Panel(
        #         accumulated_text,
        #         title=f"[bold cyan]{current_executor}[/bold cyan]",
        #         subtitle=f"[dim]Author: {author_name}[/dim]" if author_name else None,
        #         border_style="cyan",
        #         expand=False
        #     ))
        
        print()

    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())