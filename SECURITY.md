# Security policy

WeightBench runs untrusted agents inside a sandbox and scores them from a tamper-evident ledger. A weakness in that
machinery lets an agent obtain a score it did not earn, so we treat integrity bugs as security issues.

## Reporting

Please report privately; do not open a public issue or pull request for an unfixed bypass.

- Use GitHub's **private vulnerability reporting** for this repository (Security tab → "Report a vulnerability").
- Include: the affected component, the platform (Linux/bubblewrap, macOS/sandbox-exec, WSL2), the commit, the task id
  and seed if relevant, and a minimal reproduction (a scripted agent command or a work script is ideal). Say what the
  bypass achieves: a higher score, an undetected forward pass, a read of private data, and so on.
- We'll acknowledge your report as soon as we can and agree on a disclosure date with you. Fixed bypasses become
  regression tests, and we credit reporters who wish to be named.

## In scope

- **Sandbox escapes:** reading the repository, `artifacts/*/private/`, other runs, ledgers, or home-directory secrets
  from inside the agent sandbox; writing outside the workdir; executing blocked compilers or interpreters.
- **Network escapes:** reaching any host other than the allowlisted agent API and the local Weight API.
- **Metering bypasses:** running a subject model, or evaluating its layers on inputs, without a metered probe and
  without an audit flag; computing on weights outside the metered representation without a `metering_escape` or
  `raw_weight_read` signal; gradient descent that the guard does not block.
- **Guard and audit evasion:** starting Python without the runtime guard undetected, suppressing or forging runtime
  traces, or hiding flagged activity from the transcript audit.
- **Budget bypasses:** probes beyond a task's `n_forward` / `n_probe`, or using another run's budget or submission.
- **Ledger and score integrity:** editing a ledger or `result.json` so that `weightbench-verify` still passes, or a
  score that does not follow from the ledger.
- **Answer leakage:** ground truth or reference solutions recoverable from public artifacts, prompts, probe responses
  or error messages.
- **Reviewer manipulation:** agent output that gets a run cleared by `weightbench review --auto` when it should not be.

## Out of scope

- Arithmetic on public weights that is metered and flagged as designed (see SPEC §1, "Limits of the sandbox"):
  report it if detection is missing, not because the arithmetic is possible.
- Vulnerabilities in the agent CLIs themselves, in Hugging Face, or in third-party packages, unless WeightBench's
  configuration makes them exploitable.
- Attacks that need control of the host outside the sandbox (for example editing the harness code before a run).
- Denial of service against your own machine (an agent using its full time or memory budget).

## Known limits

The sandbox's guarantees and their limits are documented in [docs/SPEC.md](docs/SPEC.md) §1. The red-team checklist
is `weightbench/harness/redteam.md`, and `python -m weightbench.harness.sandbox selftest` checks isolation on your
host.
