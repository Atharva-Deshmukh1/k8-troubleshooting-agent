"""
Tool registry: maps tool names → kubectl commands.
Local parsers: turn kubectl output → plain English answers.
Local intents: detect questions we can answer without calling Gemini.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from .kubectl import CmdResult, run

# ── Allowed read-only tools the LLM can request ──────────────────────────────

TOOL_DESCRIPTIONS = """
get_pods(namespace?, all_namespaces?)           - list pods and their status
describe_pod(namespace, pod)                    - full details of one pod
get_logs(namespace, pod, tail_lines?)           - current pod logs
get_previous_logs(namespace, pod, tail_lines?)  - logs from last crashed container
get_events(namespace?, all_namespaces?)         - cluster events sorted by time
get_nodes()                                     - list all nodes
describe_node(node)                             - full details of one node
get_deployments(namespace?, all_namespaces?)    - list deployments
get_replicasets(namespace?)                     - list replica sets
get_daemonsets(namespace?)                      - list daemon sets
get_jobs(namespace?)                            - list jobs
get_cronjobs(namespace?)                        - list cron jobs
get_services(namespace?)                        - list services
get_endpoints(namespace?)                       - list endpoints (shows if service has backing pods)
get_ingress(namespace?)                         - list ingress rules
get_pv()                                        - list persistent volumes
get_pvc(namespace?, all_namespaces?)            - list persistent volume claims
get_storageclass()                              - list storage classes
get_hpa(namespace?, all_namespaces?)            - list Horizontal Pod Autoscalers
get_configmaps(namespace?)                      - list config maps
get_namespaces()                                - list all namespaces
top_pods(namespace?)                            - pod CPU/memory usage
top_nodes()                                     - node CPU/memory usage
get_serviceaccounts(namespace?)                 - list service accounts
get_rolebindings(namespace?)                    - list role bindings
get_clusterrolebindings()                       - list cluster role bindings
auth_can_i(verb, resource, namespace?, as?)     - check RBAC permission
exec_check(namespace, pod, host, port)          - test TCP connectivity from inside a pod (nc -zv host port)
""".strip()

# ── Write/read classification ─────────────────────────────────────────────────

# Phrases that are ALWAYS reads — checked first, before write keywords
_READ_OVERRIDES = (
    "what is", "what are", "show", "list", "get", "check", "how many",
    "is ", "are ", "why", "describe", "explain", "status",
    "logs of", "log of", "last lines", "give last", "show last",
    "tail", "fetch logs", "print logs", "give me logs",
    "last 1", "last 2", "last 3", "last 4", "last 5",
    "last 6", "last 7", "last 8", "last 9", "last 10",
    "last 20", "last 50", "last 100",
)

# Verbs that signal a change/mutation
WRITE_KEYWORDS = (
    "delete ", "create ", "apply ", "scale ", "restart ", "patch ",
    "rollout ", "replace ", "cordon", "uncordon", "drain ", "taint ",
    "label node", "annotate ",
    "deploy ", "deploy a", "add a ", "add deployment", "add pod",
    "remove pod", "remove deployment",
    "expose deployment", "expose service",
    "increase replica", "decrease replica", "set replica",
    "increase min", "increase max", "decrease min", "decrease max",
    "set min", "set max", "set desired",
    "change replica", "change min", "change max",
    "update replica", "update min", "update max",
)


def is_write_request(question: str) -> bool:
    low = question.lower()
    # Read phrases always win — checked first
    if any(kw in low for kw in _READ_OVERRIDES):
        return False
    return any(w in low for w in WRITE_KEYWORDS)


# ── Build kubectl commands from tool name + params ────────────────────────────

def build_command(name: str, params: dict, namespace: str, tail_lines: int) -> tuple[str, list[str]] | None:
    """Return (title, args) or None if tool is unknown/unsafe."""
    ns = str(params.get("namespace") or namespace)
    tail_val = params.get("tail_lines") or tail_lines
    try:
        tail_val = int(tail_val)
    except (TypeError, ValueError):
        tail_val = tail_lines
    tail = f"--tail={min(max(tail_val, 1), 500)}"
    all_ns = bool(params.get("all_namespaces"))

    def ns_args() -> list[str]:
        return ["-A"] if all_ns else ["-n", ns]

    match name:
        case "get_pods":
            return "pods", ["kubectl", "get", "pods", "-o", "wide"] + ns_args()
        case "describe_pod":
            pod = params.get("pod", "")
            if not pod:
                return None
            return f"describe {pod}", ["kubectl", "describe", "pod", pod, "-n", ns]
        case "get_logs":
            pod = params.get("pod", "")
            if not pod:
                return None
            return f"logs {pod}", ["kubectl", "logs", pod, "-n", ns, tail]
        case "get_previous_logs":
            pod = params.get("pod", "")
            if not pod:
                return None
            return f"previous logs {pod}", ["kubectl", "logs", pod, "-n", ns, "--previous", tail]
        case "get_events":
            return "events", ["kubectl", "get", "events", "--sort-by=.lastTimestamp"] + ns_args()
        case "get_nodes":
            return "nodes", ["kubectl", "get", "nodes", "-o", "wide"]
        case "describe_node":
            node = params.get("node", "")
            if not node:
                return None
            return f"describe node {node}", ["kubectl", "describe", "node", node]
        case "get_deployments":
            return "deployments", ["kubectl", "get", "deployment", "-o", "wide"] + ns_args()
        case "get_replicasets":
            return "replicasets", ["kubectl", "get", "rs", "-n", ns]
        case "get_daemonsets":
            return "daemonsets", ["kubectl", "get", "ds", "-n", ns]
        case "get_jobs":
            return "jobs", ["kubectl", "get", "jobs", "-n", ns]
        case "get_cronjobs":
            return "cronjobs", ["kubectl", "get", "cronjobs", "-n", ns]
        case "get_services":
            return "services", ["kubectl", "get", "svc", "-n", ns, "-o", "wide"]
        case "get_endpoints":
            return "endpoints", ["kubectl", "get", "endpoints", "-n", ns]
        case "get_ingress":
            return "ingress", ["kubectl", "get", "ingress", "-n", ns]
        case "get_pv":
            return "persistent volumes", ["kubectl", "get", "pv", "-o", "wide"]
        case "get_pvc":
            return "persistent volume claims", ["kubectl", "get", "pvc", "-o", "wide"] + ns_args()
        case "get_storageclass":
            return "storage classes", ["kubectl", "get", "storageclass"]
        case "get_hpa":
            return "hpa", ["kubectl", "get", "hpa", "-o", "wide"] + ns_args()
        case "get_configmaps":
            return "configmaps", ["kubectl", "get", "configmap", "-n", ns]
        case "get_namespaces":
            return "namespaces", ["kubectl", "get", "namespaces"]
        case "top_pods":
            return "pod metrics", ["kubectl", "top", "pods", "-n", ns]
        case "top_nodes":
            return "node metrics", ["kubectl", "top", "nodes"]
        case "get_serviceaccounts":
            return "service accounts", ["kubectl", "get", "serviceaccount", "-n", ns]
        case "get_rolebindings":
            return "role bindings", ["kubectl", "get", "rolebinding", "-n", ns]
        case "get_clusterrolebindings":
            return "cluster role bindings", ["kubectl", "get", "clusterrolebinding"]
        case "auth_can_i":
            verb = params.get("verb", "")
            resource = params.get("resource", "")
            if not verb or not resource:
                return None
            args = ["kubectl", "auth", "can-i", verb, resource]
            if params.get("namespace"):
                args += ["-n", ns]
            if params.get("as"):
                args.append(f"--as={params['as']}")
            return f"auth can-i {verb} {resource}", args
        case "exec_check":
            pod = params.get("pod", "")
            host = params.get("host", "")
            port = str(params.get("port", ""))
            if not pod or not host or not port:
                return None
            return (
                f"connectivity {host}:{port}",
                ["kubectl", "exec", pod, "-n", ns, "--", "nc", "-zv", "-w", "5", host, port],
            )
        case _:
            return None


def run_tools(tool_list: list[dict], namespace: str, tail_lines: int) -> list[CmdResult]:
    """Run a list of tool requests, skip duplicates."""
    results = []
    seen: set[tuple] = set()
    for tool in tool_list:
        name = tool.get("name", "")
        params = tool.get("params") or {}
        built = build_command(name, params, namespace, tail_lines)
        if not built:
            continue
        title, args = built
        key = tuple(args)
        if key in seen:
            continue
        seen.add(key)
        results.append(run(title, args))
    return results


# ── Local fast-path parsers ───────────────────────────────────────────────────

def _lines(output: str) -> list[str]:
    return [l for l in output.splitlines() if l.strip()]


def _ready_full(ready: str) -> bool:
    if "/" not in ready:
        return False
    a, b = ready.split("/", 1)
    return a == b


def local_nodes(output: str) -> str:
    lines = _lines(output)
    if len(lines) <= 1:
        return "Could not read node info from kubectl."
    ready, not_ready, details = 0, 0, []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        name, status = parts[0], parts[1]
        roles = parts[2] if len(parts) > 2 else "-"
        age = parts[3] if len(parts) > 3 else "-"
        version = parts[4] if len(parts) > 4 else "-"
        icon = "✅" if status == "Ready" else "❌"
        if status == "Ready":
            ready += 1
        else:
            not_ready += 1
        details.append(f"  {icon} {name}  status={status}  roles={roles}  age={age}  version={version}")
    headline = "✅ All nodes are ready." if not_ready == 0 else f"⚠️  {not_ready} node(s) are NOT ready."
    return f"{headline}\n\nTotal: {ready + not_ready} nodes ({ready} ready, {not_ready} not ready)\n\n" + "\n".join(details)


def local_pods(output: str, list_mode: bool = False) -> str:
    lines = _lines(output)
    if len(lines) <= 1:
        return "No pods found."
    unhealthy, warnings, rows, total = [], [], [], 0
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        total += 1
        name, ready, status, restarts, age = parts[0], parts[1], parts[2], parts[3], parts[4]
        restart_count = int(restarts.split("(")[0]) if restarts[0].isdigit() else 0

        if list_mode:
            if status in {"Running", "Completed"} and _ready_full(ready) and restart_count == 0:
                icon = "✅"
            elif restart_count > 0:
                icon = "⚠️ "
            else:
                icon = "❌"
            rows.append(f"  {icon} {name:<55} {status:<20} ready={ready}  restarts={restarts}  age={age}")
        else:
            if status not in {"Running", "Completed"}:
                unhealthy.append(f"  ❌ {name}: {status} (ready={ready})")
            elif status == "Running" and not _ready_full(ready):
                unhealthy.append(f"  ❌ {name}: not fully ready ({ready})")
            elif restart_count > 0:
                warnings.append(f"  ⚠️  {name}: has {restarts} restarts")

    if list_mode:
        return f"Pods in namespace ({total} total):\n\n" + "\n".join(rows)
    if unhealthy:
        return f"❌ {len(unhealthy)} pod(s) have problems out of {total}:\n" + "\n".join(unhealthy)
    if warnings:
        return f"⚠️  All {total} pods running, but {len(warnings)} have restarted:\n" + "\n".join(warnings)
    return f"✅ All {total} pods are healthy and running."


def local_node_metrics(output: str) -> str:
    lines = _lines(output)
    if len(lines) <= 1:
        return "Metrics not available. Is metrics-server installed?\nInstall: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append(f"  {parts[0]:<45} CPU={parts[1]:<8} ({parts[2]:<6})  Memory={parts[3]:<10} ({parts[4]})")
    return f"Node resource usage ({len(rows)} nodes):\n\n" + "\n".join(rows)


def local_pod_metrics(output: str) -> str:
    lines = _lines(output)
    if len(lines) <= 1:
        return "Pod metrics not available. Is metrics-server installed?"
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            rows.append(f"  {parts[0]:<55} CPU={parts[1]:<8}  Memory={parts[2]}")
    return f"Pod resource usage ({len(rows)} pods):\n\n" + "\n".join(rows)


def local_nodegroups(output: str) -> str:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return "Could not parse node JSON."
    label_keys = (
        "eks.amazonaws.com/nodegroup",
        "alpha.eksctl.io/nodegroup-name",
        "cloud.google.com/gke-nodepool",
        "agentpool",
        "kops.k8s.io/instancegroup",
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for item in data.get("items", []):
        name = item.get("metadata", {}).get("name", "?")
        labels = item.get("metadata", {}).get("labels", {})
        for key in label_keys:
            if labels.get(key):
                groups[labels[key]].append(name)
                break
    if not groups:
        return "No node groups found (no standard nodegroup labels on nodes)."
    lines = [
        f"  {g}: {len(nodes)} node(s)  →  {', '.join(nodes[:3])}{'...' if len(nodes) > 3 else ''}"
        for g, nodes in sorted(groups.items())
    ]
    return f"Found {len(groups)} node group(s):\n\n" + "\n".join(lines)


def local_services(output: str) -> str:
    lines = _lines(output)
    if len(lines) <= 1:
        return "No services found."
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append(f"  {parts[0]:<40} type={parts[1]:<12} cluster-ip={parts[2]:<16} port(s)={parts[4]}")
    return f"Services ({len(rows)} found):\n\n" + "\n".join(rows)


def local_deployments(output: str) -> str:
    lines = _lines(output)
    if len(lines) <= 1:
        return "No deployments found."
    unhealthy, healthy = [], []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        name, ready = parts[0], parts[1]
        age = parts[4] if len(parts) > 4 else "-"
        if "/" in ready:
            r, t = ready.split("/")
            if r != t:
                unhealthy.append(f"  ❌ {name}: {ready} ready  age={age}")
            else:
                healthy.append(f"  ✅ {name}: {ready} ready  age={age}")
        else:
            healthy.append(f"  - {name}: ready={ready}  age={age}")
    headline = f"✅ All {len(healthy)} deployments are healthy." if not unhealthy else f"⚠️  {len(unhealthy)} deployment(s) not fully ready."
    return f"{headline}\n\n" + "\n".join(unhealthy + healthy)


# ── Extract pod name and tail count from free-text questions ─────────────────

_POD_NAME_RE = re.compile(
    r"\b([a-z0-9][a-z0-9-]{3,}(?:-[a-z0-9]+){1,})\b"
)
_TAIL_RE = re.compile(r"\b(?:last|tail)\s+(\d+)\b", re.IGNORECASE)


def extract_pod_name(question: str) -> str | None:
    """Pick the longest token that looks like a k8s pod name."""
    noise = {
        "running", "ready", "status", "restarts", "error", "pending", "failed",
        "logs", "lines", "give", "show", "last", "tail", "fetch", "print",
        "of", "for", "from", "the", "pod", "container", "namespace",
    }
    candidates = [m.group(1) for m in _POD_NAME_RE.finditer(question.lower())]
    candidates = [c for c in candidates if c not in noise and len(c) > 6]
    return max(candidates, key=len) if candidates else None


def extract_tail(question: str, default: int) -> int:
    m = _TAIL_RE.search(question)
    if m:
        return min(int(m.group(1)), 500)
    return default


# ── Intent detection ──────────────────────────────────────────────────────────
# Format: intent → (keyword_list, base_kubectl_args, list_mode_flag)
# list_mode only applies to pods intent.

_LOCAL_INTENTS: dict[str, tuple[list[str], list[str], bool]] = {
    # nodes
    "nodes_health":  (["are nodes", "nodes healthy", "node status", "nodes ready"],
                      ["kubectl", "get", "nodes", "-o", "wide"], False),
    "nodes_list":    (["list nodes", "show nodes", "how many nodes", "get nodes", "all nodes"],
                      ["kubectl", "get", "nodes", "-o", "wide"], False),
    "node_metrics":  (["node cpu", "node memory", "nodes cpu", "nodes memory", "top nodes",
                       "node usage", "nodes using", "nodes resource", "whats my nodes",
                       "what's my nodes", "nodes usage"],
                      ["kubectl", "top", "nodes"], False),
    # pods
    "pods_health":   (["are pods", "pods healthy", "pods running", "pod status", "pods ok", "pods fine"],
                      ["kubectl", "get", "pods", "-o", "wide"], False),
    "pods_list":     (["list pods", "show pods", "get pods", "how many pods", "all pods", "running pods"],
                      ["kubectl", "get", "pods", "-o", "wide"], True),
    "pod_metrics":   (["pod cpu", "pod memory", "pods cpu", "pods memory", "top pods",
                       "pod usage", "pods using", "pods resource"],
                      ["kubectl", "top", "pods"], False),
    # node groups
    "nodegroups":    (["node group", "nodegroup", "node pool", "how many nodegroup", "list nodegroup"],
                      ["kubectl", "get", "nodes", "-o", "json"], False),
    # cluster-wide
    "events":        (["events", "recent events", "what happened", "cluster events", "any events"],
                      ["kubectl", "get", "events", "--sort-by=.lastTimestamp"], False),
    "namespaces":    (["namespaces", "list namespace", "show namespace", "all namespaces", "how many namespace"],
                      ["kubectl", "get", "namespaces"], False),
    # storage
    "pv":            (["persistent volume", " pv ", "list pv", "show pv"],
                      ["kubectl", "get", "pv", "-o", "wide"], False),
    "pvc":           (["pvc", "persistent volume claim", "list pvc"],
                      ["kubectl", "get", "pvc", "-o", "wide"], False),
    # services / networking
    "services":      (["services", "list service", "show service", "get service", " svc"],
                      ["kubectl", "get", "svc", "-o", "wide"], False),
    # deployments
    "deployments":   (["deployments", "list deployment", "show deployment", "get deployment"],
                      ["kubectl", "get", "deployment", "-o", "wide"], False),
}

# Intents that need -n <namespace>
_NS_SCOPED = {"pods_health", "pods_list", "pod_metrics", "events", "pvc", "services", "deployments"}

# ── Special local intents that need dynamic args ──────────────────────────────
# These need the pod name / tail from the question, so they're handled separately.

def detect_log_intent(question: str, default_tail: int) -> tuple[str, list[str]] | None:
    """
    Detect 'show me logs of <pod>' style questions.
    Returns (pod_name, kubectl_args) or None.
    """
    low = question.lower()
    log_triggers = (
        "logs of", "log of", "logs for", "log for",
        "last lines of", "last lines for",
        "give last", "show last", "show logs", "get logs",
        "fetch logs", "print logs", "give me logs",
        "last ", "tail ",          # "last 20 lines of pod-xyz"
    )
    if not any(t in low for t in log_triggers):
        return None
    pod = extract_pod_name(question)
    if not pod:
        return None
    tail = extract_tail(question, default_tail)
    return pod, ["kubectl", "logs", pod, f"--tail={tail}"]


def detect_describe_intent(question: str) -> tuple[str, str, list[str]] | None:
    """
    Detect 'describe pod <name>' or 'describe node <name>'.
    Returns (kind, name, kubectl_args) or None.
    """
    low = question.lower()
    if "describe" not in low:
        return None
    if "node" in low:
        # try to extract a node name (ip-... or hostname)
        m = re.search(r"\b(ip-[\d-]+\.[a-z0-9.]+|[a-z0-9][a-z0-9.-]{4,})\b", low)
        if m:
            name = m.group(1)
            return "node", name, ["kubectl", "describe", "node", name]
    # try pod name
    pod = extract_pod_name(question)
    if pod:
        return "pod", pod, ["kubectl", "describe", "pod", pod]
    return None


def detect_local_intent(question: str) -> tuple[str, list[str], bool] | None:
    """
    Return (intent, args, list_mode) if we can answer locally, else None.
    Does NOT handle log/describe — those are detected separately in agent.py.
    """
    low = question.lower()
    for intent, (keywords, args, list_mode) in _LOCAL_INTENTS.items():
        if any(kw in low for kw in keywords):
            return intent, args, list_mode
    return None


def answer_locally(intent: str, result: CmdResult, list_mode: bool = False) -> str:
    out = result.output
    match intent:
        case "nodes_health" | "nodes_list":
            return local_nodes(out)
        case "node_metrics":
            return local_node_metrics(out)
        case "pods_health":
            return local_pods(out, list_mode=False)
        case "pods_list":
            return local_pods(out, list_mode=True)
        case "pod_metrics":
            return local_pod_metrics(out)
        case "nodegroups":
            return local_nodegroups(out)
        case "events":
            return f"Recent cluster events:\n\n{out or 'No events found.'}"
        case "namespaces":
            ls = _lines(out)
            ns_lines = [f"  {l.split()[0]}" for l in ls[1:] if l.split()]
            return f"Namespaces ({len(ns_lines)} found):\n\n" + "\n".join(ns_lines)
        case "pv":
            return f"Persistent volumes:\n\n{out or 'None found.'}"
        case "pvc":
            return f"Persistent volume claims:\n\n{out or 'None found.'}"
        case "services":
            return local_services(out)
        case "deployments":
            return local_deployments(out)
        case _:
            return out