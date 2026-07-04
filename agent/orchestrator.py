# Import Libraries
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from config.settings import DEEPSEEK_API_KEY
from agent.tools import retrieve_recommendation
from agent.prompts import system_prompt

TOOLS = [retrieve_recommendation]

llm = ChatDeepSeek(model='deepseek-chat', api_key=DEEPSEEK_API_KEY,
                   temperature=0.3)

llm_with_tools = llm.bind_tools(TOOLS)

# Agent Node
def agent_node(state: MessagesState):
    messages = state['messages']
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt())] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# Build Graph
def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node('agent', agent_node)
    graph.add_node('tools', ToolNode(TOOLS))

    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent', tools_condition)
    graph.add_edge('tools', 'agent')
    return graph.compile()

app = build_graph()
