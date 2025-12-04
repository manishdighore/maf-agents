# Microsoft Agent Framework - Quick Reference

## 🎯 What's Included

### Scripts
1. **`1_azure_openai_chat_agent.py`** - Basic chat agent with function tools
2. **`2_magentic_orchestration.py`** - Multi-agent orchestration
3. **`3_agent_as_tool.py`** - Hierarchical agent architecture

### Features
✅ Rich terminal output with colors and formatting
✅ Streaming responses for all agent interactions
✅ Environment variable configuration via `.env`
✅ Pretty-printed agent events with panels
✅ Azure OpenAI integration

## ⚡ Quick Start

```powershell
# Run setup script (installs packages, creates .env)
.\setup.ps1

# Edit .env with your Azure OpenAI credentials
notepad .env

# Login to Azure
az login

# Run examples
python 1_azure_openai_chat_agent.py
python 2_magentic_orchestration.py
python 3_agent_as_tool.py
```

## 📋 Environment Variables (.env)

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-10-21
```

## 🎨 Rich Output Features

All scripts now include:
- **Colored output** - Different colors for different event types
- **Markdown rendering** - Automatically renders markdown in agent responses
- **Event panels** - Beautiful panels showing agent activity
- **Streaming display** - Real-time token streaming

## 🔧 print_agent_event() Function

Custom event handler that provides:
- Inline streaming for delta events
- Panel display for complete messages
- Markdown rendering support
- Color-coded by event type:
  - 🔵 Cyan: Agent messages
  - 🟣 Magenta: Orchestrator messages
  - 🟢 Green: Final results

## 📝 Pattern Summaries

### Pattern 1: Basic Chat Agent
```python
agent = AzureOpenAIChatClient(credential=credential).create_agent(
    instructions="You are a helpful assistant.",
    tools=[function1, function2]
)

async for event in agent.run_stream("Hello!"):
    print_agent_event(event)
```

### Pattern 2: Magentic Orchestration
```python
workflow = (
    MagenticBuilder()
    .participants(agent1=agent1, agent2=agent2)
    .on_event(on_event, mode=MagenticCallbackMode.STREAMING)
    .with_standard_manager(chat_client=client)
    .build()
)

async for event in workflow.run_stream(task):
    print_agent_event(event)
```

### Pattern 3: Agent as Tool
```python
# Convert agent to tool
tool = specialist_agent.as_tool(
    name="tool_name",
    description="What it does",
    arg_name="input",
    arg_description="Input description"
)

# Use in coordinator
coordinator = ChatAgent(
    name="coordinator",
    chat_client=client,
    tools=[tool]
)

async for event in coordinator.run_stream(task):
    print_agent_event(event)
```

## 🐛 Troubleshooting

### Import Errors
```powershell
uv pip install agent-framework --prerelease=allow --force-reinstall
uv pip install python-dotenv rich
```

### Authentication Errors
```powershell
az login
az account show
```

### Missing .env
```powershell
cp .env.sample .env
notepad .env
```

## 📚 Key Dependencies

- `agent-framework` - Microsoft Agent Framework
- `azure-identity` - Azure authentication
- `python-dotenv` - Environment variable loading
- `rich` - Beautiful terminal output

## 🎓 Learn More

- [MAF Documentation](https://learn.microsoft.com/en-us/agent-framework/)
- [Azure OpenAI Chat Agents](https://learn.microsoft.com/en-us/agent-framework/user-guide/agents/agent-types/azure-openai-chat-completion-agent)
- [Magentic Orchestration](https://learn.microsoft.com/en-us/agent-framework/user-guide/workflows/orchestrations/magentic)
