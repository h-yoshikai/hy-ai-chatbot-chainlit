from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

import chainlit as cl

from agent import MCPAgent
from dotenv import load_dotenv
import os

agent: StateGraph = None

load_dotenv()
api_key =os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# create agent asyncronously because getting MCP tools is asyncronous
async def create_agent():
    global agent
    agent = await MCPAgent.create(llm)

@cl.on_chat_start
async def on_chat_start():
    global agent
    # start chat after creating a global agent
    if agent is None:
        await create_agent()

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
        async with cl.Step(name="Tool", type="tool", parent_id=final_step.id) as tool_step:
            async with cl.Step(name="LLM", type="llm", parent_id=final_step.id) as normal_step:
                # invoke llm
                async for msg, metadata in agent.astream({"messages": history}, stream_mode="messages", config=RunnableConfig(**config)):
                    node = metadata.get("langgraph_node")
                    if msg.content and node == "normal":
                        await normal_step.stream_token(msg.content)
                    elif msg.content and node == "tools":
                        await tool_step.stream_token(msg.content)
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
