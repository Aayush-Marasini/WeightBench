"""Tests use toy tensors only and never run a subject model or submit a remote compute job."""
import os

os.environ["WB_HEAVY"] = "local"

# Tests never call a real agent CLI; tests/test_agent_auth.py clears this to exercise the login preflight.
os.environ["WB_SKIP_AGENT_PREFLIGHT"] = "1"
