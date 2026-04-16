# Copyright 2026 Google LLC
"""CLI runner for TC-AI Nexus sample."""

from __future__ import annotations

import asyncio

import agent
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def main() -> None:
  app_name = "tc_ai_nexus"
  user_id = "cli_user"
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  runner = Runner(
      app_name=app_name,
      agent=agent.root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )
  session = await session_service.create_session(app_name=app_name, user_id=user_id)

  while True:
    prompt = input("\nAsk TC-AI Nexus (or 'exit'): ").strip()
    if prompt.lower() in {"exit", "quit"}:
      break
    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
      if event.content and event.content.parts and event.content.parts[0].text:
        print(f"\n[{event.author}] {event.content.parts[0].text}")


if __name__ == "__main__":
  asyncio.run(main())
