"""
D&D 5e Session Prep Assistant - Published Agent using Microsoft Agent Framework.

This sample connects to a published PromptAgent in Foundry Agent Service that
prepares detailed D&D 5th Edition session content: events, characters, locations,
and rules that build a dark, engaging world for the party.

The agent outputs concise DM-facing content in GitHub Flavored Markdown with:
- Vivid _read aloud_ passages (speech, sight, smell, taste, touch, sound)
- D&D 5e rules references (ability checks, saving throws, dice rolls)
- Links to dndbeyond.com where confident URLs exist
- Text-to-image prompts describing scenes when relevant

**PUBLISHED ENDPOINT**: Calls the Agent Application endpoint — a separate ARM
resource created via `--publish-agent`. This endpoint is independent of the
development project endpoint, has its own stable URL, and requires the Azure
AI User role scoped specifically to the Agent Application resource (not the
project). Callers cannot reach unpublished agents through this URL.

  Endpoint pattern:
    {project}/applications/{app}/protocols/openai/responses

**Published Agent Constraints**:
- Only the stateless Responses API (POST /responses) is supported
- No /conversations, /files, /vector_stores — client manages history
- Requires Azure AI User role on the Agent Application resource (not project)
- The agent's instructions, model, and tools are pre-configured server-side

**`--create-agent`**: Creates a new immutable PromptAgent version via
  the azure-ai-projects SDK (AIProjectClient.agents.create_version).

**`--publish-agent`**: Creates an Agent Application ARM resource and a
  managed deployment, exposing the specified agent version through the
  stable application endpoint.

Prerequisites:
- Published PromptAgent version in Microsoft Foundry
- Azure CLI authenticated (az login)
- Azure AI User role on the Agent Application resource
- Required packages: pip install azure-ai-projects azure-identity httpx

Reference:
- https://learn.microsoft.com/azure/foundry/agents/how-to/agent-applications
- https://learn.microsoft.com/azure/foundry/agents/how-to/configure-agent
"""
# pylint: disable=duplicate-code,too-many-statements,too-many-locals
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

import argparse
import asyncio
import os
import re
import time

import httpx
from agent_framework.foundry import FoundryAgent
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import AzureCliCredential

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# Default Agent Configuration
# ============================================================================

DEFAULT_PROJECT_ENDPOINT = "https://your-foundry.services.ai.azure.com/api/projects/dnd5e"
DEFAULT_AGENT_NAME = "gm-agent"
DEFAULT_AGENT_VERSION = "17"
DEFAULT_APPLICATION_NAME = "gm-agent"
DEFAULT_MODEL_DEPLOYMENT = "gpt-5.3-chat"
DEFAULT_ARM_API_VERSION = "2026-01-15-preview"

GM_AGENT_INSTRUCTIONS = """\
# D&D 5e Session Prep Assistant

Prepare detailed D&D 5th Edition session content: events, characters, \
locations, and rules that build a dark, engaging world for the party.

> [!IMPORTANT]
> Minimize lines. Only quotes and _read aloud_ passages may be expanded. \
Collapse bullet points onto single lines where possible and makes sense. \
Optimize to reduce overall length.

## Output Rules

- GitHub Flavored Markdown. Compact layout — combine closely related \
elements on one line.
- Concise DM-facing content for the requested topic.
- _Read aloud_ passages: vivid, bleak descriptions covering speech, sight, \
smell, taste, touch, and sound as appropriate. Highlight small details when \
they should draw player attention.
- Structure with headings, lists, tables, emphasis, and emojis where \
appropriate.
- Reference D&D 5e rules explicitly; highlight ability checks, saving \
throws, and dice rolls.
- When relevant, include a text-to-image prompt describing the scene.
- Link to public pages on https://dndbeyond.com when confident the URL \
exists. Paywalled content is acceptable.
- Use emojis to improve readability and understanding

### Examples

Use these as examples of how to output specific content types.

#### DM Read Aloud sections

```markdown
### 🎙️ Read Aloud 

_Neat rows of small, tidy cots line the room..._
```

#### Image Generation Prompt

```markdown
### 🎨 Text-to-Image Prompt

_A realistic fantasy photograph of a gothic castle servants' chamber..._
```

### Image Generation Prompt Guidelines

Always start with `A realistic fantasy photograph` to ensure the model generates a realistic image. \

## Input Provided

- Campaign basics: setting, theme.
- Relevant characters, scenario, and details.

## Clarification

If more information is needed for a compelling narrative, ask up to 3 \
questions.
"""


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the D&D 5e Session Prep assistant sample.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="D&D 5e Session Prep Assistant - Published Agent via Agent Framework",
        epilog="Example: python published_gm_agent.py --interactive"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=(
            "Prepare a session opening for a party of 4 level-5 adventurers "
            "arriving at a cursed fishing village on the Sword Coast at dusk. "
            "Theme: folk horror. Include a read-aloud passage and an encounter hook."
        ),
        help="The session prep request to send to the assistant"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode for a multi-turn D&D session"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses"
    )
    parser.add_argument(
        "--project-endpoint",
        type=str,
        default=None,
        help="Foundry project endpoint (overrides env var)"
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default=None,
        help="Agent name (overrides env var)"
    )
    parser.add_argument(
        "--agent-version",
        type=str,
        default=None,
        help="Agent version (overrides env var)"
    )
    parser.add_argument(
        "--create-agent",
        action="store_true",
        help="Create/update the agent in Foundry Agent Service and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model deployment name for agent creation (default: gpt-5.3-chat)"
    )
    parser.add_argument(
        "--publish-agent",
        action="store_true",
        help="Publish the agent as an Agent Application via ARM REST API"
    )
    parser.add_argument(
        "--subscription-id",
        type=str,
        default=None,
        help="Azure subscription ID (required for --publish-agent)"
    )
    parser.add_argument(
        "--resource-group",
        type=str,
        default=None,
        help="Azure resource group (required for --publish-agent)"
    )
    parser.add_argument(
        "--application-name",
        type=str,
        default=None,
        help="Agent Application name for publish and chat (defaults to agent name)"
    )
    parser.add_argument(
        "--deployment-name",
        type=str,
        default=None,
        help="Agent deployment name for --publish-agent (defaults to 'default')"
    )
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> tuple[str, str, str]:
    """
    Resolve configuration from args, environment variables, or defaults.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (project_endpoint, agent_name, agent_version).
    """
    project_endpoint = (
        args.project_endpoint
        or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
        or DEFAULT_PROJECT_ENDPOINT
    )
    agent_name = (
        args.agent_name
        or os.environ.get("FOUNDRY_AGENT_NAME")
        or DEFAULT_AGENT_NAME
    )
    agent_version = (
        args.agent_version
        or os.environ.get("FOUNDRY_AGENT_VERSION")
        or DEFAULT_AGENT_VERSION
    )
    return project_endpoint, agent_name, agent_version


def parse_endpoint_url(project_endpoint: str) -> tuple[str, str]:
    """
    Extract account name and project name from a Foundry project endpoint URL.

    Args:
        project_endpoint: URL like
            https://<account>.services.ai.azure.com/api/projects/<project>

    Returns:
        Tuple of (account_name, project_name).

    Raises:
        ValueError: If the URL does not match the expected pattern.
    """
    match = re.match(
        r"https://([^.]+)\.services\.ai\.azure\.com/api/projects/([^/]+)",
        project_endpoint,
    )
    if not match:
        raise ValueError(
            f"Cannot parse endpoint URL: {project_endpoint}\n"
            "Expected: https://<account>.services.ai.azure.com/api/projects/<project>"
        )
    return match.group(1), match.group(2)


def publish_agent_application(
    project_endpoint: str,
    agent_name: str,
    agent_version: str,
    subscription_id: str,
    resource_group: str,
    application_name: str,
    deployment_name: str,
) -> None:
    """
    Publish an agent version as an Agent Application via the ARM REST API.

    Creates (or updates) an Agent Application ARM resource and a managed
    deployment referencing the specified agent version. The application
    provides a stable, externally-shareable endpoint with independent RBAC
    and a dedicated Entra agent identity separate from the project identity.

    After publishing, grant Azure AI User role on the Agent Application
    resource to any caller that needs to invoke the published endpoint.
    Re-assign RBAC for any tool resources the agent accesses, as the new
    agent identity differs from the shared project identity.

    Reference:
        https://learn.microsoft.com/azure/foundry/agents/how-to/agent-applications

    Args:
        project_endpoint: Foundry project endpoint URL.
        agent_name: Name of the agent to publish.
        agent_version: Immutable agent version to deploy.
        subscription_id: Azure subscription ID.
        resource_group: Resource group containing the Foundry resource.
        application_name: Name for the Agent Application ARM resource.
        deployment_name: Name for the agent deployment.
    """
    account_name, project_name = parse_endpoint_url(project_endpoint)
    credential = AzureCliCredential()
    token = credential.get_token("https://management.azure.com/.default")

    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json",
    }

    arm_base = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices"
        f"/accounts/{account_name}"
        f"/projects/{project_name}"
    )
    api_version = DEFAULT_ARM_API_VERSION

    # Step 1: Create or update the Agent Application
    app_url = f"{arm_base}/applications/{application_name}?api-version={api_version}"
    app_body = {
        "properties": {
            "agents": [{"agentName": agent_name}],
        }
    }

    print("  Step 1: Creating Agent Application...")
    with httpx.Client(timeout=120) as client:
        resp = client.put(app_url, headers=headers, json=app_body)
        if resp.status_code not in (200, 201):
            print(f"  ERROR: Failed to create application (HTTP {resp.status_code})")
            print(f"  {resp.text}")
            return
        print(f"  Application '{application_name}' created/updated.")

        # Step 2: Create or update the deployment
        deploy_url = (
            f"{arm_base}/applications/{application_name}"
            f"/agentdeployments/{deployment_name}?api-version={api_version}"
        )
        deploy_body = {
            "properties": {
                "displayName": f"{agent_name} deployment",
                "deploymentType": "Managed",
                "protocols": [
                    {"protocol": "Responses", "version": "1.0"},
                ],
                "agents": [
                    {
                        "agentName": agent_name,
                        "agentVersion": agent_version,
                    },
                ],
            }
        }

        print(f"  Step 2: Creating deployment '{deployment_name}'...")
        resp = client.put(deploy_url, headers=headers, json=deploy_body)
        if resp.status_code not in (200, 201):
            print(f"  ERROR: Failed to create deployment (HTTP {resp.status_code})")
            print(f"  {resp.text}")
            return
        print(f"  Deployment '{deployment_name}' created/updated.")

        # Step 3: Wait for deployment to reach running state
        print("  Step 3: Verifying deployment status...")
        for _ in range(12):
            resp = client.get(deploy_url, headers=headers)
            if resp.status_code == 200:
                state = resp.json().get("properties", {}).get("provisioningState", "")
                if state.lower() in ("succeeded", "running"):
                    print(f"  Deployment state: {state}")
                    break
                if state.lower() == "failed":
                    print("  ERROR: Deployment failed.")
                    print(f"  {resp.text}")
                    return
                print(f"  Deployment state: {state} (waiting...)")
            time.sleep(5)

    app_endpoint = (
        f"{project_endpoint}/applications/{application_name}"
        f"/protocols/openai"
    )
    print()
    print("  Published successfully!")
    print(f"  Application: {application_name}")
    print(f"  Deployment:  {deployment_name}")
    print(f"  Agent:       {agent_name} (v{agent_version})")
    print(f"  Endpoint:    {app_endpoint}")
    print()
    print("  NEXT: Grant 'Azure AI User' role on the Agent Application resource")
    print("        to any caller that needs to invoke the published endpoint.")


def create_agent_version(
    project_endpoint: str,
    agent_name: str,
    model: str,
) -> None:
    """
    Create a new version of the GM agent in Foundry Agent Service.

    Uses the azure-ai-projects SDK to deploy a PromptAgent with the D&D 5e
    Session Prep instructions. Each call creates a new immutable version.

    Args:
        project_endpoint: Foundry project endpoint URL.
        agent_name: Name of the agent to create/update.
        model: Model deployment name (e.g., 'gpt-5.3-chat').
    """
    credential = AzureCliCredential()
    client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )

    definition = PromptAgentDefinition(
        model=model,
        instructions=GM_AGENT_INSTRUCTIONS,
    )

    result = client.agents.create_version(
        agent_name=agent_name,
        definition=definition,
        description="D&D 5e Session Prep Assistant - prepares session content "
        "with dark atmosphere, vivid read-aloud passages, and D&D 5e rules.",
    )

    print(f"  Agent created successfully!")
    print(f"  Name:    {result.name}")
    print(f"  Version: {result.version}")
    print(f"  ID:      {result.id}")
    print(f"  Model:   {model}")


async def run_single_question(
    args: argparse.Namespace,
    project_endpoint: str,
    application_name: str,
) -> None:
    """
    Run a single session prep request against the published Agent Application.

    Uses FoundryAgent with the full project endpoint
    (https://<account>.services.ai.azure.com/api/projects/<project>) and the
    application name. The agent must already be published via --publish-agent
    before calling this function.

    Args:
        args: Parsed command-line arguments.
        project_endpoint: Full Foundry project endpoint URL.
        application_name: Name of the published Agent Application.
    """
    credential = AzureCliCredential()
    use_stream = not args.no_stream

    async with FoundryAgent(
        project_endpoint=project_endpoint,
        agent_name=application_name,
        credential=credential,
        allow_preview=True,
    ) as agent:
        print(f"\nDM Request: {args.question}")
        print()

        if use_stream:
            print("Session Prep:\n", flush=True)
            async for chunk in agent.run(args.question, stream=True):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
        else:
            result = await agent.run(args.question)
            print(f"Session Prep:\n\n{result}")


async def run_interactive(
    project_endpoint: str,
    application_name: str,
) -> None:
    """
    Run an interactive D&D 5e session prep conversation via the published
    Agent Application.

    Uses FoundryAgent with the full project endpoint. Each turn is stateless
    from the service perspective — the application endpoint does not persist
    conversation history between requests.

    Args:
        project_endpoint: Full Foundry project endpoint URL.
        application_name: Name of the published Agent Application.
    """
    credential = AzureCliCredential()

    async with FoundryAgent(
        project_endpoint=project_endpoint,
        agent_name=application_name,
        credential=credential,
        allow_preview=True,
    ) as agent:
        print("\n=== D&D 5e SESSION PREP - INTERACTIVE ===")
        print("Type 'quit' to end. Each request is stateless (history not retained).")
        print("Provide campaign context, scenario details, or ask for specific content.\n")

        while True:
            try:
                request = input("DM Request: ").strip()
                if request.lower() in ["quit", "exit", "q", ""]:
                    print("\n[Session ended.]")
                    break

                print("\nSession Prep:\n", flush=True)
                async for chunk in agent.run(request, stream=True):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                print("\n")

            except KeyboardInterrupt:
                print("\n\n[Session interrupted.]")
                break


async def main() -> None:
    """Main entry point for the D&D 5e Session Prep published agent sample."""
    args = parse_arguments()
    project_endpoint, agent_name, agent_version = get_config(args)

    if args.create_agent:
        model = (
            args.model
            or os.environ.get("FOUNDRY_MODEL_DEPLOYMENT")
            or DEFAULT_MODEL_DEPLOYMENT
        )
        print()
        print("=" * 60)
        print("  D&D 5e SESSION PREP - Create Agent Version")
        print("=" * 60)
        print()
        print(f"  Project:  {project_endpoint}")
        print(f"  Agent:    {agent_name}")
        print(f"  Model:    {model}")
        print()
        create_agent_version(project_endpoint, agent_name, model)
        return

    if args.publish_agent:
        subscription_id = (
            args.subscription_id
            or os.environ.get("AZURE_SUBSCRIPTION_ID")
        )
        resource_group = (
            args.resource_group
            or os.environ.get("AZURE_RESOURCE_GROUP")
        )
        if not subscription_id or not resource_group:
            print(
                "ERROR: --subscription-id and --resource-group are required "
                "for --publish-agent (or set AZURE_SUBSCRIPTION_ID and "
                "AZURE_RESOURCE_GROUP env vars)."
            )
            return
        application_name = (
            args.application_name
            or os.environ.get("FOUNDRY_APPLICATION_NAME")
            or agent_name
        )
        deployment_name = args.deployment_name or "default"

        print()
        print("=" * 60)
        print("  D&D 5e SESSION PREP - Publish Agent Application")
        print("=" * 60)
        print()
        print(f"  Project:      {project_endpoint}")
        print(f"  Agent:        {agent_name} (v{agent_version})")
        print(f"  Application:  {application_name}")
        print(f"  Deployment:   {deployment_name}")
        print()
        publish_agent_application(
            project_endpoint=project_endpoint,
            agent_name=agent_name,
            agent_version=agent_version,
            subscription_id=subscription_id,
            resource_group=resource_group,
            application_name=application_name,
            deployment_name=deployment_name,
        )
        return

    application_name = (
        args.application_name
        or os.environ.get("FOUNDRY_APPLICATION_NAME")
        or agent_name
    )

    print()
    print("=" * 60)
    print("  D&D 5e SESSION PREP - Published Agent Application")
    print("=" * 60)
    print()
    print(f"  Project:      {project_endpoint}")
    print(f"  Application:  {application_name}")
    print(f"  Endpoint:     {project_endpoint}/applications/{application_name}/protocols/openai")
    print()

    if args.interactive:
        await run_interactive(project_endpoint, application_name)
    else:
        await run_single_question(args, project_endpoint, application_name)


if __name__ == "__main__":
    asyncio.run(main())
