from langchain_core.messages import HumanMessage

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from state import AgentState

class MCPAgent:
    def __init__(self, llm, tools):
        self.model = llm
        self.tools = tools
        self.tool_model = llm.bind_tools(tools)
        self.agent = self._create_agent()

    @classmethod
    async def create(self, llm):
        tools = await self._get_tools()
        return self(llm, tools).agent

    def _create_agent(self) -> StateGraph:
        builder = StateGraph(AgentState)
        tool_node = ToolNode(tools=self.tools)
        builder.add_node("normal", self._call_model)
        builder.add_node("tools", tool_node)
        builder.add_node("final", self._call_final_model)

        builder.add_edge(START, "normal")
        builder.add_conditional_edges(
            "normal",
            tools_condition,
            {"tools": "tools", "__end__": "final"}
        )
        builder.add_edge("tools", "normal")
        builder.add_edge("final", END)

        return builder.compile()

    def _call_model(self, state: AgentState):
        system_prompt = {
            "role": "system",
            "content": "過去の会話は参考にしてください。必ず一番最後のユーザーの発言（質問や依頼）に主に答えてください。"
        }
        messages = [system_prompt] + state["messages"]
        response = self.tool_model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    def _call_final_model(self, state: AgentState):
        messages = state["messages"]
        last_ai_message = messages[-1]
        response = self.model.invoke(
            [
                SystemMessage("これを孫悟空の話し方で書き直してください。"),
                HumanMessage(last_ai_message.content),
            ]
        )
        # overwrite the last AI message from the agent
        response.id = last_ai_message.id
        return {"messages": [response]}

    @staticmethod
    async def _get_tools():
        client = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8001/sse/",
                    "transport": "sse",
                }
            }
        )
        tools = await client.get_tools()
        return tools
