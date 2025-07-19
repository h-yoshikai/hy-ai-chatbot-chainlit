from langchain_core.messages import HumanMessage

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState

class SimpleAgent:
    def __init__(self, llm):
        self.model = llm
        self.agent = self._create_agent()

    @classmethod
    def create(self, llm):
        return self(llm).agent

    def _create_agent(self) -> StateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("normal", self._call_model)
        builder.add_node("final", self._call_final_model)

        builder.add_edge(START, "normal")
        builder.add_edge("normal", "final")
        builder.add_edge("final", END)

        return builder.compile()

    def _call_model(self, state: AgentState):
        system_prompt = {
            "role": "system",
            "content": "過去の会話は参考にしてください。必ず一番最後のユーザーの発言（質問や依頼）に主に答えてください。"
        }
        messages = [system_prompt] + state["messages"]
        response = self.model.invoke(messages)

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
