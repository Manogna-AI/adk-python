# Copyright 2026 Google LLC
"""Gradio UI for TC-AI Nexus."""

from __future__ import annotations

import asyncio

import gradio as gr
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.genai import types

import agent

APP_NAME = "tc_ai_nexus"
USER_ID = "gradio_user"

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()
runner = Runner(
    app_name=APP_NAME,
    agent=agent.root_agent,
    artifact_service=artifact_service,
    session_service=session_service,
)


async def _ask_adk(prompt: str, session_id: str) -> str:
  user_msg = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
  text_chunks = []
  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session_id,
      new_message=user_msg,
  ):
    if event.content and event.content.parts and event.content.parts[0].text:
      text_chunks.append(event.content.parts[0].text)
  return "\n".join(text_chunks).strip()


def respond(message: str, history: list[tuple[str, str]], session_id: str):
  if not message.strip():
    return history, session_id
  answer = asyncio.run(_ask_adk(message, session_id=session_id))
  history.append((message, answer or "No response generated."))
  return history, session_id


async def _create_session() -> str:
  session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
  return session.id


def main() -> None:
  default_session = asyncio.run(_create_session())

  with gr.Blocks(title="TC-AI Nexus") as demo:
    gr.Markdown(
        """
# TC-AI Nexus
Role-oriented Teamcenter multi-agent assistant (ADK + GPT-5.3 + RAG + MCP-ready)
"""
    )
    state_session = gr.State(value=default_session)
    chatbot = gr.Chatbot(label="Nexus Conversation", height=520)
    textbox = gr.Textbox(label="Ask a Teamcenter engineering question")
    submit = gr.Button("Send")
    clear = gr.Button("Clear")

    submit.click(
        respond,
        inputs=[textbox, chatbot, state_session],
        outputs=[chatbot, state_session],
    )
    textbox.submit(
        respond,
        inputs=[textbox, chatbot, state_session],
        outputs=[chatbot, state_session],
    )

    def reset_chat():
      return [], asyncio.run(_create_session())

    clear.click(reset_chat, outputs=[chatbot, state_session])

  demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
  main()
