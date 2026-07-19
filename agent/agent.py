import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

from .tools import run_tool, write_report
from .prompts import SYSTEM_PROMPT

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.1-flash-lite"  # confirm this matches what's live in AI Studio for your key

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_web",
                description="Search the web for information. Returns titles, URLs, and snippets.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific. One query at a time."
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.FunctionDeclaration(
                name="read_page",
                description="Fetch and read the full text of a web page by URL.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to read."
                        }
                    },
                    "required": ["url"]
                }
            ),
            types.FunctionDeclaration(
                name="write_report",
                description="Write and save the final research report. Call ONLY when you have thoroughly researched the topic with at least 3 sources.",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The full Markdown report content with citations."
                        }
                    },
                    "required": ["content"]
                }
            ),
        ]
    )
]


@dataclass
class RunResult:
    """Everything we want to know about one agent run, for eval purposes.

    Kept separate from the print() statements — those are for a human
    watching the terminal live; this is for a machine analyzing many
    runs later.
    """
    question: str
    success: bool                  # did it produce a report at all?
    filename: str = ""
    iterations_used: int = 0
    searches_done: int = 0
    pages_read: int = 0
    nudge_fired: bool = False      # did Plan A (nudge) trigger?
    fallback_fired: bool = False   # did Plan B (fallback save) trigger?
    hit_max_iterations: bool = False
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def log_run(result: RunResult, log_path: str = "evals/run_logs.jsonl") -> None:
    """Append one run's result as a single JSON line.

    JSON Lines (.jsonl) means one JSON object per line — not one big JSON
    array. This lets us keep appending forever without ever re-parsing or
    rewriting the whole file, and lets the eval runner just read it line
    by line later.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")



    """Pull all function calls out of a Gemini response."""
    tool_calls = []
    if not response.candidates:
        return tool_calls
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if part.function_call and part.function_call.name:
                tool_calls.append(part.function_call)
    return tool_calls


def extract_text(response) -> str:
    """Pull plain text out of a Gemini response, if any."""
    chunks = []
    if not response.candidates:
        return ""
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if part.text:
                chunks.append(part.text)
    return "\n".join(chunks)


def _accumulate_tokens(response, result: RunResult) -> None:
    """Add this response's token usage onto the running total for the run."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    result.input_tokens += usage.prompt_token_count or 0
    result.output_tokens += usage.candidates_token_count or 0


def run_agent(question: str, max_iterations: int = 15) -> RunResult:
    print(f"\n🔍 Research question: {question}")
    print("─" * 60)

    start_time = time.monotonic()
    result = RunResult(question=question, success=False)

    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    iteration = 0
    response = chat.send_message(question)
    _accumulate_tokens(response, result)
    already_nudged = False

    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Iteration {iteration}]")

        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            stray_text = extract_text(response)

            # Plan A: nudge once — the model may just need a push to call
            # write_report instead of answering in plain text.
            if not already_nudged:
                already_nudged = True
                result.nudge_fired = True
                print("⚠️  No tool call. Nudging the model to use write_report...")
                response = chat.send_message(
                    "You responded with plain text instead of calling a tool. "
                    "If you have gathered enough information, call write_report now "
                    "with your full Markdown report as the content argument. "
                    "Do not repeat the answer as plain text."
                )
                _accumulate_tokens(response, result)
                continue

            # Plan B: safety net — if the model still won't call the tool but
            # it clearly did real research and gave a real answer, save that
            # answer ourselves rather than throwing away good work.
            if stray_text and result.searches_done > 0:
                result.fallback_fired = True
                print("⚠️  Model still didn't call write_report. Saving its last answer as a fallback report.")
                write_result = write_report(content=stray_text, question=question)
                result.iterations_used = iteration
                result.duration_seconds = time.monotonic() - start_time
                if write_result.get("success"):
                    result.success = True
                    result.filename = write_result["filename"]
                    print(f"  ✅ Fallback report saved: {result.filename}")
                    log_run(result)
                    return result
                else:
                    result.error = write_result.get("error", "unknown write_report error")
                    print(f"  ❌ Fallback save failed: {result.error}")

            print("⚠️  Agent stopped without writing a report and nothing usable to salvage.")
            print("--- What Gemini actually said ---")
            print(stray_text or "(no text either)")
            print("----------------------------------")
            if not result.error:
                result.error = "No tool call and nothing salvageable"
            break

        result_parts = []
        report_filename = ""

        for fc in tool_calls:
            tool_name = fc.name
            tool_input = dict(fc.args) if fc.args else {}

            print(f"  → Tool: {tool_name}({str(tool_input)[:80]})")

            if tool_name == "write_report":
                write_result = write_report(
                    content=tool_input.get("content", ""),
                    question=question
                )
                result_str = str(write_result)
                report_filename = write_result.get("filename", "")
                if write_result.get("success"):
                    print(f"  ✅ Report saved: {report_filename}")
                else:
                    print(f"  ❌ write_report failed: {write_result.get('error')}")
            else:
                result_str = run_tool(tool_name, tool_input)
                if tool_name == "search_web":
                    result.searches_done += 1
                elif tool_name == "read_page":
                    result.pages_read += 1

            result_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result_str}
                )
            )

        response = chat.send_message(result_parts)
        _accumulate_tokens(response, result)

        if report_filename:
            result.iterations_used = iteration
            result.duration_seconds = time.monotonic() - start_time
            result.success = True
            result.filename = report_filename
            print(f"\n✅ Done in {iteration} iterations.")
            log_run(result)
            return result

    print(f"\n⚠️  Hit max iterations ({max_iterations}).")
    result.iterations_used = iteration
    result.duration_seconds = time.monotonic() - start_time
    result.hit_max_iterations = True
    if not result.error:
        result.error = "Hit max iterations without producing a report"
    log_run(result)
    return result
