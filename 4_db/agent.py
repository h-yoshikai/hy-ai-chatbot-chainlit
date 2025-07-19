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
        builder.add_node("prediction", self._call_prediction_model)
        builder.add_node("final", self._call_final_model)

        builder.add_edge(START, "normal")
        builder.add_conditional_edges(
            "normal",
            tools_condition,
            {"tools": "tools", "__end__": "prediction"}
        )
        builder.add_edge("tools", "normal")
        builder.add_edge("prediction", "final")
        builder.add_edge("final", END)

        return builder.compile()

    def _call_model(self, state: AgentState):
        system_prompt = {
            "role": "system",
            "content": "過去の会話は参考にしてください。必ず一番最後のユーザーの発言（質問や依頼）に主に答えてください。"
        }
        messages = [system_prompt] + state["messages"]
        response = self.tool_model.invoke(messages)

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

    def _call_prediction_model(self, state: AgentState):
        system_prompt = {
            "role": "system",
            "content": "一番最後のAIの回答に対して、現在ユーザが興味を持っていると思われる次の質問をユーザ目線で3つ予測してください。それに伴い、過去の会話は参考にしてください。3つの質問は改行して出力しますが、番号付けなどはしなくて良いです。"
        }
        messages = [system_prompt] + state["messages"]
        try:
            response = self.model.invoke(messages)
            return {"predicted_questions": response.content}
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise

    @staticmethod
    async def _get_tools():
        client = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://host.docker.internal:8001/sse/",
                    "transport": "sse",
                }
            }
        )
        tools = await client.get_tools()
        return tools
