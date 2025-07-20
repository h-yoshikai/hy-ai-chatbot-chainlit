from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

import chainlit as cl

from agent import SimpleAgent
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
agent = SimpleAgent.create(llm)

@cl.on_message
async def on_message(msg: cl.Message):
    # get a user message
    user_input = msg.content

    # get a chat history and add the latest user message to it
    history = cl.user_session.get("history", [])
    history.append(HumanMessage(content=user_input))

    config = {"configurable": {"thread_id": cl.context.session.id}}
    final_answer = cl.Message(content="")
    ai_content = ""

    async with cl.Step(name="Agent", type="agent") as final_step:
        async with cl.Step(name="LLM", type="llm", parent_id=final_step.id) as normal_step:
            # invoke llm
            for msg, metadata in agent.stream({"messages": history}, stream_mode="messages", config=RunnableConfig(**config)):
                node = metadata.get("langgraph_node")
                if msg.content and node == "normal":
                    await normal_step.stream_token(msg.content)
                elif (
                    msg.content
                    and not isinstance(msg, HumanMessage)
                    and node == "final"
                ):
                    ai_content += msg.content
                    await final_answer.stream_token(msg.content)
                    await final_step.stream_token(msg.content)

    # add an AI's answer to history
    if ai_content:
        history.append(AIMessage(content=ai_content))
    cl.user_session.set("history", history)

    await final_answer.send()
