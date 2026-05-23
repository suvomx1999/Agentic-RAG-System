# LangGraph Framework

## What is LangGraph?

LangGraph is a library built on top of LangChain for creating stateful, multi-actor applications with LLMs. It models the application as a graph where nodes represent computation steps and edges represent transitions between steps. LangGraph supports cycles, branching, and conditional routing, making it ideal for building complex agent workflows.

## Core Concepts

### StateGraph
The StateGraph is the primary building block. It defines:
- A state schema (typically a TypedDict) that flows through the graph
- Nodes that transform the state
- Edges (fixed and conditional) that determine the flow

### State
State in LangGraph is a TypedDict or Pydantic model that:
- Gets passed to every node function
- Is updated by node return values (partial updates merged into state)
- Supports reducer functions for list fields (e.g., appending to message lists)
- Persists across graph execution steps

### Nodes
Nodes are Python functions that:
- Accept the current state as input
- Return a dictionary of state updates (partial update pattern)
- Can call external APIs, tools, or other services
- Should be idempotent when possible for reliability

### Edges
LangGraph supports several edge types:
- **Normal edges**: Fixed transitions from one node to another
- **Conditional edges**: Dynamic routing based on state values
- **Entry edges**: Define the starting node of the graph

### Conditional Edges
Conditional edges use router functions that:
- Accept the current state
- Return the name of the next node (or END to terminate)
- Use Literal type hints for graph visualization
- Cannot modify state — they are pure routing decisions

## Building a Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Literal

class AgentState(TypedDict):
    query: str
    answer: Optional[str]
    iterations: int

def process_query(state: AgentState) -> dict:
    return {"answer": "processed", "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> Literal["process", "__end__"]:
    if state["iterations"] >= 3:
        return "__end__"
    return "process"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("process", process_query)
graph.set_entry_point("process")
graph.add_conditional_edges("process", should_continue)
app = graph.compile()
```

## Streaming

LangGraph supports multiple streaming modes:
- **values**: Stream the full state after each node execution
- **updates**: Stream only the state updates from each node
- **events**: Stream detailed events including LLM tokens

### Event Streaming
```python
async for event in app.astream_events(input, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        # Handle streaming tokens
        pass
```

## Persistence and Checkpointing

LangGraph supports state persistence through checkpointers:
- MemorySaver for development
- SQLiteSaver for production
- Enables conversation history and state recovery
- Supports time-travel debugging (replay from any checkpoint)

## Error Handling

Best practices for error handling in LangGraph:
- Use try/except within node functions
- Set error fields in state rather than raising exceptions
- Implement fallback nodes for graceful degradation
- Use max_iterations guards to prevent infinite loops
- Apply retry logic with exponential backoff for API calls

## Common Patterns

### Agentic Loop
The most common LangGraph pattern is the agentic loop:
1. Agent analyzes the query and decides on an action
2. Action is executed (tool call, retrieval, etc.)
3. Result is evaluated
4. If satisfactory, return answer; otherwise, loop back to step 1

### Human-in-the-Loop
LangGraph supports human intervention points:
- Interrupt the graph at specific nodes
- Wait for human approval or input
- Resume execution with human-provided data

### Subgraphs
Complex workflows can be composed using subgraphs:
- Each subgraph is a self-contained StateGraph
- Subgraphs can be nested within parent graphs
- State mapping between parent and child graphs
