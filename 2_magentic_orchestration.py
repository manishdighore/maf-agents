"""
Magentic Orchestration Example
Demonstrates multi-agent collaboration with dynamic coordination
"""
import asyncio
import logging
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from agent_framework import (
    ChatAgent,
    MagenticAgentDeltaEvent,
    MagenticAgentMessageEvent,
    MagenticBuilder,
    MagenticFinalResultEvent,
    MagenticOrchestratorMessageEvent
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_agent_event(event):
    """Pretty-print MAF agent events with colors and source information."""
    event_type_str = type(event).__name__
    content = None
    border_style = 'white'
    title = None
    
    if isinstance(event, MagenticAgentDeltaEvent):
        # Streaming tokens from agent (print inline, no panel)
        print(event.text, end="", flush=True)
        return
        
    elif isinstance(event, MagenticAgentMessageEvent):
        # Complete agent responses
        msg = getattr(event, 'message', None)
        agent_id = getattr(event, 'agent_id', 'unknown')
        if msg is not None:
            response_text = getattr(msg, 'text', '')
            role = getattr(msg, 'role', None)
            role_str = role.value if role else 'unknown'
            content = f"**Agent:** {agent_id}\n**Role:** {role_str}\n\n{response_text}"
            border_style = 'cyan'
    
    elif isinstance(event, MagenticOrchestratorMessageEvent):
        # Orchestrator messages
        text = getattr(event.message, 'text', '')
        kind = getattr(event, 'kind', 'unknown')
        content = f"**Kind:** {kind}\n\n{text}"
        border_style = 'magenta'
    
    elif isinstance(event, MagenticFinalResultEvent):
        # Final result
        msg = getattr(event, 'message', None)
        if msg:
            text = getattr(msg, 'text', '')
            content = f"**Final Result**\n\n{text}"
            border_style = 'green'
    
    if content:
        if title is None:
            title = f"[bold {border_style}]{event_type_str}[/bold {border_style}]"
        
        try:
            if isinstance(content, str) and any(md_marker in content for md_marker in ['#', '```', '*', '-', '>', '**']):
                panel_content = Markdown(content)
            else:
                panel_content = str(content)
        except:
            panel_content = str(content)
        
        rprint(Panel(
            panel_content,
            title=title,
            border_style=border_style,
            expand=False
        ))
        rprint()


async def main() -> None:
    print("=" * 60)
    print("Magentic Orchestration Demo")
    print("=" * 60)
    
    # Create Azure OpenAI client
    credential = AzureCliCredential()
    
    # Define specialized agents
    researcher_agent = ChatAgent(
        name="ResearcherAgent",
        description="Specialist in research and information gathering",
        instructions=(
            "You are a Researcher. You find information and provide factual analysis "
            "without complex computation. Focus on gathering and presenting information clearly."
        ),
        chat_client=AzureOpenAIChatClient(credential=credential),
    )
    
    analyst_agent = ChatAgent(
        name="AnalystAgent",
        description="Specialist in data analysis and interpretation",
        instructions=(
            "You are an Analyst. You analyze data, identify patterns, and provide insights. "
            "Break down complex information into clear, actionable conclusions."
        ),
        chat_client=AzureOpenAIChatClient(credential=credential),
    )
    
    writer_agent = ChatAgent(
        name="WriterAgent",
        description="Specialist in content creation and documentation",
        instructions=(
            "You are a Writer. You create well-structured, clear, and engaging content. "
            "Synthesize information into polished documents and reports."
        ),
        chat_client=AzureOpenAIChatClient(credential=credential),
    )
    
    # State for streaming callback
    last_stream_agent_id: str | None = None
    stream_line_open: bool = False
    
    # Unified callback for all events
    async def on_event(event) -> None:
        nonlocal last_stream_agent_id, stream_line_open
        
        if isinstance(event, MagenticAgentDeltaEvent):
            if last_stream_agent_id != event.agent_id or not stream_line_open:
                if stream_line_open:
                    print()
                rprint(f"\n[cyan][{event.agent_id}]:[/cyan] ", end="", flush=True)
                last_stream_agent_id = event.agent_id
                stream_line_open = True
        
        elif isinstance(event, MagenticAgentMessageEvent):
            if stream_line_open:
                print()
                stream_line_open = False
        
        elif isinstance(event, MagenticFinalResultEvent):
            if stream_line_open:
                print()
                stream_line_open = False
        
        print_agent_event(event)
    
    def on_exception(exception: Exception) -> None:
        print(f"Exception occurred: {exception}")
        logger.exception("Workflow exception", exc_info=exception)
    
    # Build the workflow
    print("\nBuilding Magentic Workflow with 3 specialized agents...")
    
    workflow = (
        MagenticBuilder()
        .participants(
            researcher=researcher_agent,
            analyst=analyst_agent,
            writer=writer_agent
        )
        .with_standard_manager(
            chat_client=AzureOpenAIChatClient(credential=credential),
            max_round_count=8,
            max_stall_count=3,
            max_reset_count=2,
        )
        .build()
    )
    
    # Define the task
    task = (
        "Research the top 3 programming languages in 2025, analyze their key strengths "
        "and use cases, and write a concise summary report (max 300 words) comparing them. "
        "Include a recommendation for which language to learn for web development."
    )
    
    print(f"\nTask: {task}")
    print("\nStarting workflow execution...\n")
    
    # Run the workflow
    try:
        completion_event = None
        async for event in workflow.run_stream(task):
            await on_event(event)
            if isinstance(event, MagenticFinalResultEvent):
                completion_event = event
        
        if completion_event is not None:
            data = getattr(completion_event, "data", None)
            preview = getattr(data, "text", None) or (str(data) if data is not None else "")
            print(f"\n\nWorkflow completed successfully!")
    
    except Exception as e:
        print(f"\nWorkflow execution failed: {e}")
        logger.exception("Workflow exception", exc_info=e)


if __name__ == "__main__":
    asyncio.run(main())
