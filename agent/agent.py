"""
Main agent brain.
Routing: local fast-path → LLM plan → LLM diagnose → guarded write.
Keeps conversation memory (pod names, namespace, previous findings).
Every action asks permission before running.
Prints raw Gemini output so user sees what's happening.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass, field

from .config import load_settings
from .kubectl import CmdResult, format_evidence, kubectl_available, run
from .tools import (
    TOOL_DESCRIPTIONS, _NS_SCOPED,
    answer_locally, build_command,
    detect_describe_intent, detect_local_intent, detect_log_intent,
    extract_pod_name, is_write_request, run_tools,
)


# ── Terminal helpers ──────────────────────────────────────────────────────────

def ask(prompt: str) -> bool:
    """Ask yes/no. Only 'y' / 'yes' returns True."""
    try:
        ans = input(f"\n{prompt} [yes/no]: ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def ask_typed(prompt: str, expected: str) -> bool:
    """Dangerous actions: user must type the exact string to confirm."""
    print(f"\n⚠️  {prompt}")
    try:
        ans = input(f"Type  '{expected}'  to confirm, or press Enter to cancel: ").strip()
    except EOFError:
        return False
    return ans == expected


def print_step(title: str, body: str = "") -> None:
    bar = "─" * max(0, 50 - len(title))
    print(f"\n── {title} {bar}")
    if body:
        print(body)


def print_answer(text: str) -> None:
    bar = "=" * 52
    print(f"\n{bar}\n{text}\n{bar}")


def print_raw_gemini(label: str, raw: str) -> None:
    """Show the raw LLM response so users can see what Gemini said."""
    bar = "·" * 52
    print(f"\n{bar}")
    print(f"  Gemini → {label}")
    print(bar)
    # Trim very long responses for readability
    preview = raw.strip()
    if len(preview) > 1800:
        preview = preview[:1800] + "\n  ... (truncated)"
    print(preview)
    print(bar)


# ── Conversation memory ───────────────────────────────────────────────────────

@dataclass
class Memory:
    last_pods: list[str] = field(default_factory=list)
    last_namespace: str = ""
    chat_history: list[dict] = field(default_factory=list)

    def note_pods(self, text: str) -> None:
        noise = {
            "running", "ready", "status", "restarts", "error", "pending",
            "failed", "logs", "lines", "namespace", "deployment", "service",
        }
        found = re.findall(r"\b([a-z0-9][a-z0-9-]{3,}(?:-[a-z0-9]+){1,})\b", text.lower())
        self.last_pods = [p for p in found if p not in noise][:5]

    def note_answer(self, question: str, answer: str) -> None:
        self.note_pods(question + " " + answer)
        self.chat_history.append({"q": question, "a": answer[:400]})
        if len(self.chat_history) > 6:
            self.chat_history.pop(0)

    def context_for_llm(self) -> str:
        if not self.chat_history:
            return "(none)"
        return "\n\n".join(
            f"Q: {t['q']}\nA: {t['a']}" for t in self.chat_history[-3:]
        )

    def enrich_question(self, question: str) -> str:
        low = question.lower()
        if self.last_pods and re.search(r"\b(it|that pod|the pod|this pod)\b", low):
            question = f"{question}  [referring to pod: {self.last_pods[0]}]"
        return question


# ── Gemini caller ─────────────────────────────────────────────────────────────

def _redact(text: str) -> str:
    patterns = [
        re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)\S+"),
        re.compile(r"(?i)(token\s*[=:]\s*)\S+"),
        re.compile(r"(?i)(password\s*[=:]\s*)\S+"),
        re.compile(r"(?i)(secret\s*[=:]\s*)\S+"),
        re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
    ]
    for p in patterns:
        text = p.sub(r"\1[REDACTED]", text)
    return text


def _call_raw(prompt: str, model: str, api_key: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
    return resp.text or ""


def call_gemini_json(label: str, prompt: str, model: str, api_key: str) -> dict:
    raw = _call_raw(prompt, model, api_key)
    print_raw_gemini(label, raw)           # ← always show raw output
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(cleaned[s:e + 1])
            except json.JSONDecodeError:
                pass
    return {"action": "final_answer", "answer": raw, "root_cause": "", "evidence": [], "fix": [], "confidence": "Low"}


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_PROMPT = """\
You are a Kubernetes operations assistant helping a beginner troubleshoot their cluster.
Choose the smallest set of safe read-only kubectl checks needed to answer the question.

RULES:
- Return ONLY valid JSON. No markdown, no text outside the JSON.
- Never request write, delete, or modify operations.
- If you already have enough evidence, use final_answer immediately.
- Use pod names and namespaces that appear in the question — do not ask for them again.
- For connectivity issues (timeout, connection refused, no route to host), use exec_check.
- For log questions, use get_logs with the exact pod name.
- For EKS nodegroup min/max: note these require `aws eks describe-nodegroup` which may not
  be available; if so, explain that in final_answer and tell the user the exact command.

Available tools:
{tools}

Default namespace: {namespace}

Recent conversation:
{context}

Question: {question}
Evidence so far: {evidence}

Return JSON:
{{
  "action": "run_tools" | "final_answer" | "ask_user",
  "reason": "one sentence",
  "tools": [{{"name": "tool_name", "params": {{"namespace": "...", "pod": "..."}}}}],
  "answer": "full answer (only when action=final_answer)",
  "question": "one short question (only when action=ask_user)"
}}
"""

DIAGNOSIS_PROMPT = """\
You are a Kubernetes expert. A beginner needs a clear answer.
Analyze ONLY the evidence below. Do not invent facts not in the evidence.

RULES:
- Return ONLY valid JSON. No markdown, no text outside the JSON.
- Plain English that a beginner understands.
- Every "evidence" bullet must cite something actually visible in the kubectl output.
- "fix" must have concrete steps — exact kubectl commands where possible.
- If the problem is network/connectivity: check service endpoints and whether target IP is routable.
- Only use run_tools if one more very specific check is needed that you haven't already run.

Available tools (only if truly needed):
{tools}

Default namespace: {namespace}

Recent conversation:
{context}

Question: {question}

Collected evidence:
{evidence}

Return JSON:
{{
  "action": "final_answer" | "run_tools" | "ask_user",
  "answer": "Yes/No/Unknown — plain English one-liner summary",
  "root_cause": "one sentence: exact reason from evidence",
  "evidence": ["exact finding from kubectl output", "..."],
  "fix": ["step 1 with exact command", "step 2", "..."],
  "confidence": "High | Medium | Low",
  "reason": "why more tools needed (only if action=run_tools)",
  "tools": [{{"name": "tool_name", "params": {{}}}}],
  "question": "one short question (only if action=ask_user)"
}}
"""

WRITE_PLAN_PROMPT = """\
You are a Kubernetes operations assistant.
The user wants to make a change. Generate the exact kubectl command or YAML manifest.

RULES:
- Return ONLY valid JSON. No markdown, no text outside the JSON.
- Prefer safe alternatives: `kubectl rollout restart` over deleting pods.
- For new deployments: generate a complete YAML manifest in the "yaml_manifest" field.
- For scaling/deleting/restarting: generate the kubectl command in "command".
- Include a dry_run_command when --dry-run=client is applicable.
- If request is too vague, set command=null and explain in warning.
- Namespace: {namespace}

User request: {question}

Return JSON:
{{
  "understood": "plain English: what the user wants",
  "command": ["kubectl", "..."] or null,
  "yaml_manifest": "full YAML string or null",
  "dry_run_command": ["kubectl", "apply", "-f", "-", "--dry-run=client"] or null,
  "warning": "any risk or important note"
}}
"""


# ── Write / mutation flow ─────────────────────────────────────────────────────

def handle_write(question: str, namespace: str, api_key: str, model: str) -> tuple[int, str]:
    if not api_key:
        print("⚠️  GEMINI_API_KEY is missing. Cannot plan write operations.")
        return 1, ""

    print_step("Change Request", f'You want to make a change:\n  "{question}"')

    if not ask("Should I plan this change with Gemini?"):
        print("OK — no changes made.")
        return 0, ""

    try:
        data = call_gemini_json(
            "write plan",
            WRITE_PLAN_PROMPT.format(question=question, namespace=namespace),
            model, api_key,
        )
    except Exception as e:
        print(f"Gemini error: {e}")
        return 1, ""

    understood = data.get("understood", question)
    command = data.get("command")
    yaml_manifest = data.get("yaml_manifest", "")
    dry_run = data.get("dry_run_command")
    warning = data.get("warning", "")

    print_step("What I understood", understood)
    if warning:
        print(f"\n⚠️  Note: {warning}")

    # ── YAML manifest path (new deployments, pods, etc.) ──────────────────
    if yaml_manifest and not command:
        print(f"\nYAML manifest I will apply:\n")
        print(yaml_manifest)

        if dry_run and ask("Run a dry-run first? (safe — no real changes)"):
            # pipe yaml into kubectl apply --dry-run
            import subprocess
            r = subprocess.run(
                ["kubectl", "apply", "-f", "-", "--dry-run=client"],
                input=yaml_manifest, text=True,
                capture_output=True,
            )
            output = (r.stdout + r.stderr).strip()
            print(f"\nDry run output:\n{output}")
            if r.returncode != 0:
                print("\n❌ Dry run failed. I will NOT apply the manifest.")
                return 1, ""

        confirm_phrase = "apply manifest"
        if not ask_typed("This will CREATE resources in your cluster.", confirm_phrase):
            print("Cancelled — no changes made.")
            return 0, ""

        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_manifest)
            tmp = f.name
        try:
            r = subprocess.run(["kubectl", "apply", "-f", tmp], capture_output=True, text=True)
            output = (r.stdout + r.stderr).strip()
            if r.returncode == 0:
                print(f"\n✅ Done!\n{output}")
                return 0, output
            else:
                print(f"\n❌ Apply failed:\n{output}")
                return 1, ""
        finally:
            os.unlink(tmp)

    # ── Direct kubectl command path ────────────────────────────────────────
    if not command:
        print("\nI cannot generate a safe command for this. Please do it manually.")
        return 0, ""

    print(f"\nCommand I want to run:\n  {' '.join(command)}")

    if dry_run and ask("Run a dry-run first? (safe — no real changes)"):
        r = run("dry run", dry_run)
        print(f"\nDry run output:\n{r.output}")
        if not r.ok():
            print("\n❌ Dry run failed. I will NOT run the real command.")
            return 1, ""

    confirm_phrase = " ".join(command[1:4])
    if not ask_typed("This will make a REAL change to your cluster.", confirm_phrase):
        print("Cancelled — no changes made.")
        return 0, ""

    result = run("write", command)
    if result.ok():
        print(f"\n✅ Done!\n{result.output}")
        return 0, result.output
    print(f"\n❌ Command failed:\n{result.output}")
    return 1, ""


# ── LLM diagnostic flow ───────────────────────────────────────────────────────

def run_llm_flow(
    question: str,
    namespace: str,
    tail_lines: int,
    api_key: str,
    model: str,
    memory: Memory,
) -> tuple[int, str]:
    if not api_key:
        print("⚠️  GEMINI_API_KEY is missing. Add it to your .env file.")
        return 1, ""

    MAX_ROUNDS = 3
    evidence_parts: list[str] = []
    enriched = memory.enrich_question(question)

    for round_num in range(1, MAX_ROUNDS + 1):
        evidence = "\n\n".join(evidence_parts) if evidence_parts else "(none)"

        # ── Planning call ──────────────────────────────────────────────────
        print_step(f"Planning  [{round_num}/{MAX_ROUNDS}]", "Figuring out which checks to run...")

        if not ask("OK to call Gemini to plan the checks?"):
            print("Stopped — no LLM calls made.")
            return 0, ""

        try:
            plan = call_gemini_json(
                "plan",
                PLANNER_PROMPT.format(
                    tools=TOOL_DESCRIPTIONS,
                    namespace=namespace,
                    question=enriched,
                    evidence=_redact(evidence),
                    context=memory.context_for_llm(),
                ),
                model, api_key,
            )
        except Exception as e:
            print(f"Gemini error: {e}")
            return 1, ""

        action = str(plan.get("action", "")).lower()

        if action == "ask_user":
            q = plan.get("question", "Can you give more details?")
            print(f"\nGemini needs more info:\n  {q}")
            try:
                clarification = input("Your answer: ").strip()
            except EOFError:
                clarification = ""
            enriched = f"{enriched}\nUser clarification: {clarification}"
            continue

        if action == "final_answer":
            ans = str(plan.get("answer", "(no answer)"))
            print_answer(ans)
            return 0, ans

        # action == run_tools
        tools_requested = plan.get("tools", [])
        reason = plan.get("reason", "")
        if reason:
            print(f"\nReason: {reason}")

        if not tools_requested:
            print("Gemini did not request any tools.")
            return 0, ""

        commands_preview: list[tuple[str, list[str]]] = []
        print("\nI want to run these read-only checks:")
        for tool in tools_requested:
            built = build_command(tool.get("name", ""), tool.get("params") or {}, namespace, tail_lines)
            if built:
                title, args = built
                print(f"  • {' '.join(args)}")
                commands_preview.append((title, args))

        if not commands_preview:
            print("No valid commands found in the plan.")
            return 0, ""

        if not ask("Should I run these checks?"):
            print("Stopped — no commands ran.")
            return 0, ""

        results: list[CmdResult] = []
        for title, args in commands_preview:
            print(f"  ⏳ {' '.join(args)}")
            results.append(run(title, args))

        evidence_parts.append(format_evidence(results))
        evidence = "\n\n".join(evidence_parts)

        # ── Diagnosis call ─────────────────────────────────────────────────
        print_step("Diagnosing", "Analyzing the results...")

        if not ask("OK to call Gemini to analyze the results?"):
            print("\nCollected evidence (not analyzed):\n")
            print(evidence)
            return 0, ""

        try:
            diagnosis = call_gemini_json(
                "diagnosis",
                DIAGNOSIS_PROMPT.format(
                    tools=TOOL_DESCRIPTIONS,
                    namespace=namespace,
                    question=enriched,
                    evidence=_redact(evidence),
                    context=memory.context_for_llm(),
                ),
                model, api_key,
            )
        except Exception as e:
            print(f"Gemini error: {e}")
            return 1, ""

        diag_action = str(diagnosis.get("action", "")).lower()

        # One more targeted check → run it then re-diagnose immediately
        if diag_action == "run_tools" and round_num < MAX_ROUNDS:
            extra_tools = diagnosis.get("tools", [])
            if extra_tools:
                extra_cmds: list[tuple[str, list[str]]] = []
                print("\nGemini wants one more targeted check:")
                for tool in extra_tools:
                    built = build_command(tool.get("name", ""), tool.get("params") or {}, namespace, tail_lines)
                    if built:
                        title, args = built
                        print(f"  • {' '.join(args)}")
                        extra_cmds.append((title, args))

                if extra_cmds and ask("Run this extra check?"):
                    for title, args in extra_cmds:
                        print(f"  ⏳ {' '.join(args)}")
                    extra_results = [run(t, a) for t, a in extra_cmds]
                    evidence_parts.append(format_evidence(extra_results))
                    evidence = "\n\n".join(evidence_parts)

                    # Re-diagnose with new evidence — no re-planning
                    print_step("Diagnosing", "Re-analyzing with the new results...")
                    if not ask("OK to call Gemini to analyze the results?"):
                        print("\nCollected evidence:\n")
                        print(evidence)
                        return 0, ""
                    try:
                        diagnosis = call_gemini_json(
                            "re-diagnosis",
                            DIAGNOSIS_PROMPT.format(
                                tools=TOOL_DESCRIPTIONS,
                                namespace=namespace,
                                question=enriched,
                                evidence=_redact(evidence),
                                context=memory.context_for_llm(),
                            ),
                            model, api_key,
                        )
                    except Exception as e:
                        print(f"Gemini error: {e}")
                        return 1, ""

        answer_text = _format_diagnosis(diagnosis)
        print_answer(answer_text)
        _offer_fix(diagnosis, question, namespace, api_key, model)
        return 0, answer_text

    print("Reached the maximum number of Gemini rounds.")
    return 0, ""


def _format_diagnosis(data: dict) -> str:
    parts = [data.get("answer", "Unknown")]
    if data.get("root_cause"):
        parts += ["", "Root cause:", f"  {data['root_cause']}"]
    ev = data.get("evidence", [])
    if ev:
        if isinstance(ev, str):
            ev = [ev]
        parts += ["", "Evidence:"] + [f"  • {e}" for e in ev]
    fix = data.get("fix", [])
    if fix:
        if isinstance(fix, str):
            fix = [fix]
        parts += ["", "Suggested fix:"] + [f"  • {f}" for f in fix]
    if data.get("confidence"):
        parts += ["", f"Confidence: {data['confidence']}"]
    return "\n".join(parts)


def _offer_fix(data: dict, original_question: str, namespace: str, api_key: str, model: str) -> None:
    fix = data.get("fix", [])
    if isinstance(fix, str):
        fix = [fix]
    # Only offer if fix contains a runnable kubectl command
    kubectl_fixes = [
        f for f in fix
        if f and "kubectl" in f
        and not any(skip in f.lower() for skip in ("no fix", "n/a", "manually verify"))
    ]
    if not kubectl_fixes:
        return

    print("\n💡 Gemini suggested an automated fix:")
    for item in kubectl_fixes:
        print(f"  {item}")

    if not ask("Should I apply this fix?"):
        print("OK — you can run the command above manually.")
        return

    for item in kubectl_fixes:
        m = re.search(r"(kubectl\s+\S.*)", item)
        if m:
            cmd_str = m.group(1).strip().rstrip(".")
            cmd_tokens = cmd_str.split()
            confirm_phrase = " ".join(cmd_tokens[1:4])
            if ask_typed("This will make a REAL change to your cluster.", confirm_phrase):
                result = run("fix", cmd_tokens)
                if result.ok():
                    print(f"\n✅ Done!\n{result.output}")
                else:
                    print(f"\n❌ Failed:\n{result.output}")
            else:
                print("Cancelled — no changes made.")
            return

    print("Could not parse the fix as a kubectl command. Run it manually.")


# ── Local fast-path: logs ─────────────────────────────────────────────────────

def run_local_logs(question: str, namespace: str, tail_lines: int) -> tuple[bool, str]:
    info = detect_log_intent(question, tail_lines)
    if not info:
        return False, ""
    pod, base_args = info
    args = base_args + ["-n", namespace]

    print_step("Local Log Fetch", f"I can fetch the logs directly.\n\nCommand:\n  {' '.join(args)}")
    if not ask("Should I run this?"):
        return False, ""

    result = run("logs", args)
    if not result.ok() and not result.output:
        print(f"Log fetch failed (exit {result.exit_code}):\n{result.output}")
        return False, ""

    answer = f"Logs for {pod} (last {tail_lines} lines):\n\n{result.output or '(no output)'}"
    return True, answer


# ── Local fast-path: describe ─────────────────────────────────────────────────

def run_local_describe(question: str, namespace: str) -> tuple[bool, str]:
    info = detect_describe_intent(question)
    if not info:
        return False, ""
    kind, name, base_args = info

    if kind == "pod":
        args = base_args + ["-n", namespace]
    else:
        args = base_args

    print_step("Local Describe", f"I can describe this {kind} directly.\n\nCommand:\n  {' '.join(args)}")
    if not ask("Should I run this?"):
        return False, ""

    result = run("describe", args)
    if not result.ok() and not result.output:
        print(f"Describe failed (exit {result.exit_code}):\n{result.output}")
        return False, ""

    answer = f"Description of {kind} '{name}':\n\n{result.output or '(no output)'}"
    return True, answer


# ── Local fast-path: general intents ─────────────────────────────────────────

def run_local(question: str, namespace: str, tail_lines: int) -> tuple[bool, str]:
    intent_info = detect_local_intent(question)
    if not intent_info:
        return False, ""

    intent, base_args, list_mode = intent_info
    args = list(base_args)
    if intent in _NS_SCOPED and "-n" not in args and "-A" not in args:
        args += ["-n", namespace]

    print_step("Local Check", f"I can answer this without calling Gemini.\n\nCommand:\n  {' '.join(args)}")
    if not ask("Should I run this check?"):
        return False, ""

    result = run(intent, args)
    if not result.ok():
        print(f"Command failed (exit {result.exit_code}):\n{result.output}")
        print("\nI'll try asking Gemini instead.")
        return False, ""

    answer = answer_locally(intent, result, list_mode)
    return True, answer


# ── Question router ───────────────────────────────────────────────────────────

def answer_question(
    question: str,
    namespace: str,
    tail_lines: int,
    api_key: str,
    model: str,
    memory: Memory,
) -> tuple[int, str]:

    question = memory.enrich_question(question)

    # 1. Write/change request
    if is_write_request(question):
        print_step("Change Request Detected", "This looks like a write/change operation. Handling carefully.")
        rc, ans = handle_write(question, namespace, api_key, model)
        if ans:
            memory.note_answer(question, ans)
        return rc, ans

    # 2. Log fetch (local — no LLM needed)
    answered, answer = run_local_logs(question, namespace, tail_lines)
    if answered:
        print_answer(answer)
        memory.note_answer(question, answer)
        return 0, answer

    # 3. Describe (local — no LLM needed)
    answered, answer = run_local_describe(question, namespace)
    if answered:
        print_answer(answer)
        memory.note_answer(question, answer)
        return 0, answer

    # 4. General local fast-path
    answered, answer = run_local(question, namespace, tail_lines)
    if answered:
        print_answer(answer)
        memory.note_answer(question, answer)
        return 0, answer

    # 5. LLM path
    print_step("Calling Gemini", "I need Gemini's help to answer this properly.")
    rc, ans = run_llm_flow(question, namespace, tail_lines, api_key, model, memory)
    if ans:
        memory.note_answer(question, ans)
    return rc, ans


# ── Chat loop ─────────────────────────────────────────────────────────────────

def run_chat(namespace: str, tail_lines: int, api_key: str, model: str) -> int:
    print("\n🤖  Kubernetes AI Agent")
    print(f"    Namespace : {namespace}")
    print(f"    Model     : {model}")
    print("    Type your question, or 'exit' to quit.\n")
    memory = Memory(last_namespace=namespace)

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", "bye"}:
            print("Bye!")
            return 0
        rc, _ = answer_question(question, namespace, tail_lines, api_key, model, memory)
        if rc != 0:
            print("  (something went wrong — you can keep asking)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kubernetes AI Agent — ask in plain English.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pod_agent.py "are my pods healthy?"
  python3 pod_agent.py "give last 50 lines of logs of extraction-workflow-server-abc123"
  python3 pod_agent.py "describe pod my-pod-xyz"
  python3 pod_agent.py "why is my app crashing?"
  python3 pod_agent.py "deploy a deployment named testing with 3 replicas with node affinity testing"
  python3 pod_agent.py "restart deployment my-app"
  python3 pod_agent.py --chat
  python3 pod_agent.py -n production "list pods"
        """,
    )
    parser.add_argument("question", nargs="*", help="Your question")
    parser.add_argument("--namespace", "-n", default=None, help="Kubernetes namespace")
    parser.add_argument("--tail-lines", type=int, default=None, help="Log lines to fetch (default 200)")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    settings = load_settings(args.namespace, args.tail_lines)

    if not kubectl_available():
        print("❌ kubectl not found. Install kubectl and ensure it's in your PATH.")
        return 1

    if args.chat or not args.question:
        return run_chat(settings.namespace, settings.tail_lines, settings.gemini_api_key, settings.gemini_model)

    memory = Memory(last_namespace=settings.namespace)
    rc, _ = answer_question(
        " ".join(args.question),
        settings.namespace,
        settings.tail_lines,
        settings.gemini_api_key,
        settings.gemini_model,
        memory,
    )
    return rc