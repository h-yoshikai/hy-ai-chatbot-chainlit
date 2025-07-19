from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

import asyncpg
from agent import MCPAgent
from dotenv import load_dotenv
import os

agent: StateGraph = None

load_dotenv()
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# for chat app to store message history to DB
dbhost = os.environ.get("DB_HOST")
dbuser = os.environ.get("DB_USER")
dbpassword = os.environ.get("DB_PASSWORD")
dbname = os.environ.get("DB_NAME")
conn_str = f"postgresql+asyncpg://{dbuser}:{dbpassword}@{dbhost}:5432/{dbname}"
cl_data._data_layer = SQLAlchemyDataLayer(conn_str)

# create agent asyncronously because getting MCP tools is asyncronous
async def create_agent():
    global agent
    agent = await MCPAgent.create(llm)

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User,
) -> cl.User | None:
    return default_user

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
    predicted_questions = ""

    async with cl.Step(name="Agent", type="agent") as final_step:
        async with cl.Step(name="Prediction", type="llm", parent_id=final_step.id) as prediction_step:
            async with cl.Step(name="Tool", type="tool", parent_id=final_step.id) as tool_step:
                async with cl.Step(name="LLM", type="llm", parent_id=final_step.id) as normal_step:
                    # invoke llm
                    async for msg, metadata in agent.astream({"messages": history}, stream_mode="messages", config=RunnableConfig(**config)):
                        node = metadata.get("langgraph_node")
                        if msg.content and node == "normal":
                            await normal_step.stream_token(msg.content)
                        elif msg.content and node == "tools":
                            await tool_step.stream_token(msg.content)
                        elif msg.content and node == "prediction":
                            predicted_questions += msg.content
                            await prediction_step.stream_token(msg.content)
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

    next_questions = [q.strip() for q in predicted_questions.split('\n') if q.strip()]
    actions = [
        cl.Action(
            name=f"predicted_question",
            payload={"value": q},
            label=q
        )
        for _, q in enumerate(next_questions)
    ]
    await cl.Message(content="関連", actions=actions).send()

@cl.action_callback("predicted_question")
async def on_action(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="user").send()
    # generate an answer against the action that user pressed
    await on_message(cl.Message(content=action.payload["value"]))

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    steps = thread.get("steps", [])
    history = []
    for message in steps:
        if message["type"] == "user_message":
            history.append(HumanMessage(content=message["output"]))
        else:
            history.append(AIMessage(content=message["output"]))

    cl.user_session.set("history", history)