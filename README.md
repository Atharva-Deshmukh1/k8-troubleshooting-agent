# Kubernetes AI Agent

Ask Kubernetes questions in plain English. The agent tries to answer locally first,
then calls Gemini if it needs help. Every action requires your permission.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Run

```bash
# Ask one question
python3 pod_agent.py "are my pods healthy?"
python3 pod_agent.py "why is my app crashing?"
python3 pod_agent.py "how many nodes are running?"

# Write/change actions (always asks for confirmation)
python3 pod_agent.py "delete pod my-broken-pod"
python3 pod_agent.py "restart deployment my-app"

# Chat mode (keep asking questions)
python3 pod_agent.py --chat

# Use a specific namespace
python3 pod_agent.py -n production "are pods healthy?"
```

## How It Works

```
Your question
     │
     ▼
Is it a write/change request? ──yes──► Gemini plans the command
     │                                  → Shows you the command
     │ no                               → Asks for typed confirmation
     ▼                                  → Runs it
Can I answer locally? ──yes──► Runs one kubectl command
     │                          → Shows you plain English answer
     │ no
     ▼
Calls Gemini to plan checks
     │
     ▼
Shows you which commands it wants to run → asks permission
     │
     ▼
Runs approved commands
     │
     ▼
Calls Gemini to analyze results → asks permission
     │
     ▼
Prints: Answer + Root Cause + Evidence + Fix + Confidence
```

## File Structure

```
pod_agent.py          ← entry point
agent/
  agent.py            ← main brain (routing, LLM, write actions)
  tools.py            ← tool registry + local answer parsers
  kubectl.py          ← runs shell commands
  config.py           ← loads .env settings
.env.example          ← copy to .env and fill in your key
requirements.txt
```

## Permission Prompts

Every step asks before doing anything:

- **Read checks**: `Should I run this check? [yes/no]`
- **LLM calls**: `OK to call Gemini? [yes/no]`
- **Write actions**: Must type the exact command to confirm (e.g. `delete pod my-pod`)

## What It Can Answer Locally (No Gemini Call)

- Are nodes healthy? How many nodes?
- Are pods healthy? List pods?
- Node/pod CPU and memory usage
- Node groups
- Recent cluster events
- Persistent volumes

## What Goes to Gemini

- Why is X crashing?
- Is my app working end to end?
- Anything involving logs, describe, events correlation
- Anything vague or multi-step
- All write/change operations
