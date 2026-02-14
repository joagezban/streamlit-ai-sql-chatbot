from __future__ import annotations

import copy
import json
from typing import Any, Callable, Optional

from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

def _get_endpoint_task_type(endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    w = WorkspaceClient()
    ep = w.serving_endpoints.get(endpoint_name)
    return ep.task

def is_endpoint_supported(endpoint_name: str) -> bool:
    """Check if the endpoint has a supported task type."""
    task_type = _get_endpoint_task_type(endpoint_name)
    supported_task_types = ["agent/v1/chat", "agent/v2/chat", "llm/v1/chat"]
    return task_type in supported_task_types

def _validate_endpoint_task_type(endpoint_name: str) -> None:
    """Validate that the endpoint has a supported task type."""
    if not is_endpoint_supported(endpoint_name):
        raise Exception(
            f"Detected unsupported endpoint type for this basic chatbot template. "
            f"This chatbot template only supports chat completions-compatible endpoints. "
            f"For a richer chatbot template with support for all conversational endpoints on Databricks, "
            f"see https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app"
        )

def _normalize_chat_completion_message(choice_message: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize the OpenAI-like chat completion message to a simple dict.
    Keeps tool_calls if present.
    """
    msg = dict(choice_message)  # shallow copy

    # Some models return content as a list of parts; normalize to a string.
    content = msg.get("content")
    if isinstance(content, list):
        combined = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        msg["content"] = combined

    return msg

def _query_endpoint(
    endpoint_name: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Calls a model serving endpoint. Returns a list of messages (agent schema) or a single assistant message in a list."""
    _validate_endpoint_task_type(endpoint_name)

    inputs: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
    if tools is not None:
        inputs["tools"] = tools
        if tool_choice is not None:
            inputs["tool_choice"] = tool_choice

    res = get_deploy_client("databricks").predict(
        endpoint=endpoint_name,
        inputs=inputs,
    )

    # Agent endpoints (Databricks agent framework) may return multiple messages
    if isinstance(res, dict) and "messages" in res:
        return res["messages"]

    # Chat-completions endpoints (OpenAI-like)
    if isinstance(res, dict) and "choices" in res:
        choice_message = res["choices"][0]["message"]
        return [_normalize_chat_completion_message(choice_message)]

    raise Exception(
        "This app can only run against:"
        "1) Databricks foundation model or external model endpoints with the chat task type (described in https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#chat-completion-model-query)"
        "2) Databricks agent serving endpoints that implement the conversational agent schema documented "
        "in https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent"
    )

ToolExecutor = Callable[[str, dict[str, Any]], str]
def query_endpoint(
    endpoint_name: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Any] = "auto",
    tool_executor: Optional[ToolExecutor] = None,
    max_tool_rounds: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Query a chat-completions or agent serving endpoint.

    Returns:
        tuple: (final_assistant_message, all_messages_including_tool_calls)
    """
    # Work on a copy so callers can choose whether to persist tool messages or not.
    msgs = copy.deepcopy(messages)

    for _ in range(max_tool_rounds + 1):
        returned = _query_endpoint(
            endpoint_name=endpoint_name,
            messages=msgs,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice if tools is not None else None,
        )

        # Agent endpoints may return a whole message trace; persist it.
        if len(returned) > 1:
            msgs.extend(returned)
            final_msg = returned[-1]
            return final_msg, msgs

        assistant_msg = returned[-1]
        tool_calls = assistant_msg.get("tool_calls")

        if not tool_calls:
            # Persist the final assistant message in history.
            msgs.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                }
            )
            return assistant_msg, msgs

        if tool_executor is None:
            raise Exception("Model requested tool_calls, but no tool_executor was provided.")

        msgs.append(
            {
                "role": "assistant",
                "content": assistant_msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            fn = (call.get("function") or {})
            fn_name = fn.get("name")
            raw_args = fn.get("arguments", "{}")

            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception as e:
                raise Exception(f"Failed to parse tool arguments for {fn_name}: {raw_args}") from e

            result_str = tool_executor(fn_name, args)

            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result_str,
                }
            )

    raise Exception("Exceeded max_tool_rounds without reaching a final assistant response.")