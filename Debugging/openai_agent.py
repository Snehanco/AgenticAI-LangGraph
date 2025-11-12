from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool

llm = init_chat_model("groq:openai/gpt-oss-20b")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_llm(state: State):
        return {"messages": llm.invoke(state["messages"])}

    graph_workflow.add_node("agent", call_llm)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()

    return agent


# agent = make_default_graph()


def make_alternative_graph():
    """Make a tool calling agent"""

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b

        Args:
            a: first int
            b: second int
        """
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = llm.bind_tools([add])

    def call_model(state: State):
        return {"messages": model_with_tools.invoke(state["messages"])}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    builder = StateGraph(State)

    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("agent", should_continue)

    agent = builder.compile()
    return agent


agent = make_alternative_graph()
