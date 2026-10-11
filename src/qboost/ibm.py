"""IBM Quantum access. The API key is read from a local ``.env`` file and never printed.

``.env`` (gitignored) in the project root:

    IBM_QUANTUM_TOKEN=...
    IBM_QUANTUM_INSTANCE=...   # optional: CRN or instance name
"""

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / ".env"


def load_env(path: Path = DEFAULT_ENV) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignores blanks and # comments; strips matching quotes."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def get_service(env_path: Path = DEFAULT_ENV) -> Any:
    """Authenticated QiskitRuntimeService for the IBM Quantum Platform."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    env = {**load_env(env_path), **{k: v for k, v in os.environ.items() if k.startswith("IBM_")}}
    token = env.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(f"IBM_QUANTUM_TOKEN not found in {env_path} or the environment.")
    kwargs: dict[str, Any] = {"channel": "ibm_quantum_platform", "token": token}
    if env.get("IBM_QUANTUM_INSTANCE"):
        kwargs["instance"] = env["IBM_QUANTUM_INSTANCE"]
    return QiskitRuntimeService(**kwargs)
