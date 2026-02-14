import logging
import os
import streamlit as st
from model_serving_utils import query_endpoint, is_endpoint_supported

import json
import time
import re
from databricks.sdk import WorkspaceClient

# Context for this app: this is a Databricks Streamlit chatbot app that can optionally
# query a Databricks SQL warehouse to answer user prompts. The chatbot's AI backend is not 
# OpenAI, but a Databricks model serving endpoint that uses a Meta foundation model.

# Requirements:
# It has system (developer) messages that provide basic knowledge about the databricks SQL
# warehouse database  to ensure contextual awareness: the bot knows the schema of the database.
# It responds to user questions naturally and flexibly, without relying on predefined or 
# rigid templates.
# It demonstrates the use of function calling by invoking one or more functions to perform 
# the SQL querying task. Make sure you don’t ask the LLM to generate SQL and then parse it 
# (extracting ```sql (...) ```); instead, provide the function definition directly to the LLM.

# The intended workflow is:
#   1. User asks question.
#   2. LLM itself decides whether to call the tool or not. If it decides to call it - you should call it and return result back to the same LLM
#   3. LLM gets the result with this messages array roles:
#   	a. System prompt
#   	b. User question
#   	c. Assistant (asks to call a tool)
#   	d. Tool (result  of the query)
#   4. LLM summarizes

# Function calling should work like the following:
# When you provide context to an LLM (through a system prompt and the ongoing conversation 
# between the user and the assistant), you can also include a tool payload. This payload 
# typically describes a function, the meaning could be:
# “I have a function called get_weather with the parameter country (string). The function 
# returns the weather (temperature) of the given city.”
# When a user asks a question, the LLM can decide whether to call a tool or not.
# If the LLM decides to call a tool, the name of the tool and the required parameters will 
# appear in its response.
# The application should then handle this tool call request — extract the function name and 
# parameters, execute the function, and send the results back to the LLM with the role tool.
# After receiving the result, the LLM will continue the process (for example, it might call 
# another tool or generate the final answer).

# Implementing SQL querying for prompts using DIAL GPT-5.2 and GitHub Copilot GPT-5.2

# ----------------------------
# Tool: Databricks SQL querying
# ----------------------------

SQL_WAREHOUSE_ID = os.getenv("DBSQL_WAREHOUSE_ID")
DBSQL_HTTP_PATH = os.getenv("DBSQL_HTTP_PATH")

DBSQL_CATALOG = os.getenv("DBSQL_CATALOG")
DBSQL_SCHEMA = os.getenv("DBSQL_SCHEMA")

DATABRICKS_HOST = os.getenv("DBSQL_HOST")
DATABRICKS_TOKEN = os.getenv("DBSQL_TOKEN")
DATABRICKS_CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID")
DATABRICKS_CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET")

SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')

def _warehouse_id_from_http_path(http_path: str) -> str | None:
    """
    Common http_path: /sql/1.0/warehouses/<warehouse_id>
    """
    if not http_path:
        return None
    m = re.search(r"/warehouses/([^/]+)", http_path)
    return m.group(1) if m else None

if not SQL_WAREHOUSE_ID and DBSQL_HTTP_PATH:
    SQL_WAREHOUSE_ID = _warehouse_id_from_http_path(DBSQL_HTTP_PATH)

def _get_workspace_client() -> WorkspaceClient:
    """
    Avoid configuring multiple auth methods simultaneously.
    - If OAuth env vars are present, rely on default auth resolution (do NOT pass token).
    - Else, if host+token are provided, use PAT.
    - Else, rely on default auth (Databricks Apps / CLI / instance profile).
    """
    if DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET:
        return WorkspaceClient()

    if DATABRICKS_HOST and DATABRICKS_TOKEN:
        return WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

    return WorkspaceClient()


_FORBIDDEN_SQL_TOKENS = (
    "insert", "update", "delete", "merge", "drop", "create", "alter", "truncate",
    "grant", "revoke", "vacuum", "optimize", "refresh", "set ", "use ",
)

_ALLOWED_PREFIXES = ("select", "with", "show", "describe", "desc", "explain")

def _validate_readonly_sql(sql: str) -> None:
    s = (sql or "").strip()
    if not s:
        raise ValueError("Empty SQL statement.")

    # Allow a single trailing semicolon (LLMs often add it), but reject any others.
    if s.endswith(";"):
        s = s[:-1].rstrip()

    if ";" in s:
        raise ValueError("Only a single statement is allowed (no semicolons).")

    low = s.lower().lstrip()

    if not low.startswith(_ALLOWED_PREFIXES):
        raise ValueError(f"Only read-only statements are allowed: {', '.join(_ALLOWED_PREFIXES)}")

    for tok in _FORBIDDEN_SQL_TOKENS:
        if tok in low:
            raise ValueError(f"Forbidden SQL token detected: {tok!r}")

def _execute_statement_sync(
    warehouse_id: str,
    statement: str,
    timeout_s: int = 60,
    catalog: str | None = None,
    schema: str | None = None,
) -> dict:
    """
    Execute a SQL statement and return dict with {columns: [...], rows: [...]}.

    Uses Databricks SQL Statement Execution API via databricks-sdk.
    """
    def _state_to_str(x) -> str | None:
        if x is None:
            return None
        # databricks-sdk may return enums with .value or .name
        val = getattr(x, "value", None)
        if isinstance(val, str):
            return val
        name = getattr(x, "name", None)
        if isinstance(name, str):
            return name
        # fallback: "StatementState.SUCCEEDED" -> "SUCCEEDED"
        s = str(x)
        return s.split(".")[-1] if s else None

    w = _get_workspace_client()

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        catalog=catalog,
        schema=schema,
        wait_timeout="5s",  # short wait; we'll poll if still running
    )

    statement_id = getattr(resp, "statement_id", None) or (resp.get("statement_id") if isinstance(resp, dict) else None)
    if not statement_id:
        raise RuntimeError(f"Missing statement_id in response: {resp}")

    deadline = time.time() + timeout_s
    status = None
    st_resp = None

    while time.time() < deadline:
        st_resp = w.statement_execution.get_statement(statement_id)
        status = getattr(st_resp, "status", None)
        state_name = _state_to_str(getattr(status, "state", None))

        if state_name in ("SUCCEEDED", "FAILED", "CANCELED"):
            break

        time.sleep(0.5)

    final_state = _state_to_str(getattr(status, "state", None))
    if final_state != "SUCCEEDED":
        err = getattr(status, "error", None)
        msg = getattr(err, "message", None) if err else None
        raise RuntimeError(
            f"SQL statement did not succeed (state={final_state}): {msg or ''}".strip()
        )

    chunk = w.statement_execution.get_statement_result_chunk_n(statement_id, 0)

    manifest = getattr(st_resp, "manifest", None) if st_resp else None
    schema_obj = getattr(manifest, "schema", None) if manifest else None
    cols = getattr(schema_obj, "columns", None) if schema_obj else None
    col_names = [getattr(c, "name", "") for c in (cols or [])]

    result = getattr(chunk, "data_array", None)
    rows = result if isinstance(result, list) else []

    return {"columns": col_names, "rows": rows}

def query_databricks_sql(sql: str) -> str:
    """
    Tool entrypoint. Returns JSON string:
      {"columns": [...], "rows": [[...], ...]}
    """
    if not SQL_WAREHOUSE_ID:
        raise RuntimeError(
            "SQL warehouse not configured. Set SQL_WAREHOUSE_ID (preferred) or DBSQL_HTTP_PATH."
        )

    _validate_readonly_sql(sql)
    data = _execute_statement_sync(
        SQL_WAREHOUSE_ID,
        sql,
        timeout_s=60,
        catalog=DBSQL_CATALOG,
        schema=DBSQL_SCHEMA,
    )
    return json.dumps(data)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_databricks_sql",
            "description": (
                "Run a read-only Databricks SQL (Spark SQL) query against the Databricks SQL warehouse and return results as JSON "
                "with 'columns' and 'rows'. IMPORTANT: avoid non-Databricks functions like STR_TO_DATE; use to_date/to_timestamp/"
                "try_to_timestamp/date_format as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single read-only SQL statement (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN). "
                            "Do not include semicolons. Prefer adding LIMIT (<= 200) where appropriate."
                        ),
                    }
                },
                "required": ["sql"],
            },
        },
    }
]

def tool_executor(name: str, args: dict) -> str:
    if name == "query_databricks_sql":
        return query_databricks_sql(sql=(args or {}).get("sql", ""))
    raise ValueError(f"Unknown tool: {name}")

# ----------------------------
# Databricks Streamlit app
# ----------------------------


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
assert SERVING_ENDPOINT, \
    ("Unable to determine serving endpoint to use for chatbot app. If developing locally, "
     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
     "deploying to a Databricks app, include a serving endpoint resource named "
     "'serving_endpoint' with CAN_QUERY permissions, as described in "
     "https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app#deploy-the-databricks-app")

# Check if the endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("Databricks Chatbot App with SQL Querying")

# Check if endpoint is supported and show appropriate UI
if not endpoint_supported:
    st.error("⚠️ Unsupported Endpoint Type")
    st.markdown(
        f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this basic chatbot template.\n\n"
        "This template only supports chat completions-compatible endpoints.\n\n"
        "👉 **For a richer chatbot template** that supports all conversational endpoints on Databricks, "
        "please see the [Databricks documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app)."
    )
else:
    st.markdown(
        "This is a training final project for Epam Systems's Generative AI Foundations for Data Analytics Engineers course.\n\n" \
        "Author: Geza Bankovics on January 13, 2026 using DIAL GPT-5.1, GitHub Copilot GPT-5.2 and this Databricks app template: https://github.com/databricks/app-templates/tree/main/streamlit-chatbot-app\n\n" \
        "Technology used: Databricks SQL, Databricks Model Serving and Streamlit."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful data assistant. "
                    "When the user asks questions that require database data, call the tool query_databricks_sql. "
                    "You MUST write Databricks SQL (Spark SQL) dialect. "
                    "Date/time rules: prefer year(col), month(col), date_trunc('month', col), "
                    "or EXTRACT(MONTH FROM col) / EXTRACT(YEAR FROM col) (parentheses required). "
                    "Do NOT use MySQL-only routines like STR_TO_DATE. "
                    "These are the tables and columns for the ecommerce Databricks schema:"
                    "customers (customer_id, first_name, last_name, email, country, signup_date, marketing_channel, is_active)"
                    "products (product_id, product_name, category, unit_cost, is_active)"
                    "orders (order_id, customer_id, order_date, status, total_amount)"
                    "order_items (order_id, product_id, quantity, unit_price, discount)"
                    "Use read-only SQL only. After you receive tool results, explain them clearly and concisely.\n\n"
                    "If schema is unknown, you may call SHOW TABLES or DESCRIBE <table> first."
                ),
            }
        ]

    # Display chat messages (skip system/tool for cleaner UI)
    for message in st.session_state.messages:
        if message["role"] in ("system", "tool"):
            continue

        # Skip intermediate "assistant tool call" messages (often have content=None)
        if message["role"] == "assistant" and message.get("tool_calls"):
            continue

        with st.chat_message(message["role"]):
            st.markdown(message.get("content") or "")

    if prompt := st.chat_input("Ask a question about your data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            assistant_msg, updated_messages = query_endpoint(
                endpoint_name=SERVING_ENDPOINT,
                messages=st.session_state.messages,
                max_tokens=400,
                tools=TOOLS,
                tool_choice="auto",
                tool_executor=tool_executor,
                max_tool_rounds=3,
            )
            assistant_content = assistant_msg.get("content") or ""
            st.markdown(assistant_content)

        # Replace session messages with the full conversation including tool calls
        st.session_state.messages = updated_messages
