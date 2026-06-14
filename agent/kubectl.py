"""Run kubectl commands and return output."""
from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class CmdResult:
    title: str          # human-readable label
    args: list[str]     # the actual command
    exit_code: int
    output: str

    def ok(self) -> bool:
        return self.exit_code == 0

    def display_cmd(self) -> str:
        return " ".join(self.args)


def kubectl_available() -> bool:
    return shutil.which("kubectl") is not None


def run(title: str, args: list[str], timeout: int = 45) -> CmdResult:
    """Run a shell command and return its output."""
    try:
        proc = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return CmdResult(title, args, proc.returncode, proc.stdout.strip())
    except FileNotFoundError:
        return CmdResult(title, args, 127, f"Command not found: {args[0]}")
    except subprocess.TimeoutExpired:
        return CmdResult(title, args, 124, f"Timed out after {timeout}s")


def format_evidence(results: list[CmdResult]) -> str:
    """Format command results into readable evidence text."""
    sections = []
    for r in results:
        sections.append(f"=== {r.display_cmd()} (exit={r.exit_code}) ===\n{r.output or '(no output)'}")
    return "\n\n".join(sections)
