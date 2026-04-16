# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TC-AI Nexus: Teamcenter role-oriented multi-agent framework."""

from __future__ import annotations

import json
import os
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

from rag_index import retrieve_teamcenter_knowledge


def _get_model() -> LiteLlm:
  model_name = os.getenv("TC_AI_MODEL", "openai/gpt-5.3")
  return LiteLlm(model=model_name)


def _build_mcp_toolsets() -> list[Any]:
  """Enable MCP tools via TC_AI_MCP_STDIO_JSON env variable.

  Expected format:
  [{"command":"python","args":["my_mcp_server.py"],"allow":["search_logs"]}]
  """
  raw = os.getenv("TC_AI_MCP_STDIO_JSON", "").strip()
  if not raw:
    return []

  toolsets = []
  configs = json.loads(raw)
  for cfg in configs:
    toolsets.append(
        MCPToolset(
            connection_params=StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
            ),
            tool_filter=cfg.get("allow"),
        )
    )
  return toolsets


def collect_context(question: str, role: str = "all_roles") -> str:
  """Tool used by every role agent to query Teamcenter domain knowledge."""
  return retrieve_teamcenter_knowledge(question=question, role=role)


model = _get_model()
common_tools = [collect_context, *_build_mcp_toolsets()]

planning_agent = LlmAgent(
    name="planning_agent",
    model=model,
    description="Decomposes complex Teamcenter tasks into actionable plan steps.",
    instruction="""
You are the Planning Agent in TC-AI Nexus.
Break user problems into a precise role-oriented execution plan with sections:
1) Objective
2) Assumptions
3) Step-by-step execution plan
4) Risks and checks
5) Required role agents
Always ask for missing environment details if confidence is low.
""",
    output_key="execution_plan",
    tools=common_tools,
)

server_side_agent = LlmAgent(
    name="server_side_agent",
    model=model,
    description="Specialist for ITK, SOA, handlers, and server customizations.",
    instruction="""
You are the server-side Teamcenter specialist.
Use collect_context with role='server_side_developer' for every recommendation.
Provide code-level diagnostics, impact analysis, and remediation steps.
Never claim execution happened unless user confirms it.
""",
    tools=common_tools,
)

bmide_agent = LlmAgent(
    name="bmide_agent",
    model=model,
    description="Specialist for BMIDE schema, templates, and deployment deltas.",
    instruction="""
You are the BMIDE modeling specialist.
Use collect_context with role='bmide_modeler' for every recommendation.
Highlight migration risks and rollback checkpoints.
""",
    tools=common_tools,
)

ui_agent = LlmAgent(
    name="ui_agent",
    model=model,
    description="Specialist for Active Workspace UI configuration and conditions.",
    instruction="""
You are the Teamcenter UI specialist.
Use collect_context with role='ui_engineer' for every recommendation.
Produce declarative configuration guidance and validation checklist.
""",
    tools=common_tools,
)

devops_agent = LlmAgent(
    name="devops_agent",
    model=model,
    description="Specialist for CI/CD, observability, and Teamcenter deployment ops.",
    instruction="""
You are the DevOps specialist.
Use collect_context with role='devops_engineer' for every recommendation.
Provide deployment-safe runbooks with rollback and SLO verifications.
""",
    tools=common_tools,
)

prod_support_agent = LlmAgent(
    name="prod_support_agent",
    model=model,
    description="Specialist for incident triage and root-cause workflows.",
    instruction="""
You are the production support specialist.
Use collect_context with role='production_support' for every recommendation.
Always produce triage timeline, probable root causes, and verification steps.
""",
    tools=common_tools,
)

orchestrator_agent = LlmAgent(
    name="orchestrator_agent",
    model=model,
    description=(
        "Routes requests to planning and specialist role agents in a "
        "human-in-the-loop workflow."
    ),
    instruction="""
You are the TC-AI Nexus Orchestrator Agent.

Operating model:
- First, invoke planning_agent for a structured execution plan.
- Then delegate to one or more role agents based on the request context.
- Merge outputs into a single response with these sections:
  A) Plan
  B) Role-specific recommendations
  C) Human validation checklist (must be explicit)
  D) Safe execution order

Mandatory rules:
- Maintain human control: no irreversible action without human approval.
- Reference the plan from planning_agent in final answer.
- If MCP tools are available, mention which diagnostics should be run.
""",
    sub_agents=[
        planning_agent,
        server_side_agent,
        bmide_agent,
        ui_agent,
        devops_agent,
        prod_support_agent,
    ],
)

# ADK runtime expects this symbol.
root_agent = orchestrator_agent
