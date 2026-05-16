# Security Policy

## Supported Versions

CryoSentinel is in early-stage release. Only the latest minor release receives security updates.

| Version | Supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no |

## Reporting a Vulnerability

If you believe you have found a security vulnerability in CryoSentinel — including but not limited to:

- A path-traversal or remote-code-execution vector in the inference scripts or Modal entrypoints
- A way to trigger arbitrary network calls via a maliciously-crafted config file
- A bug that allows reading credentials from environment variables or service-account JSON files
- A supply-chain concern with a pinned dependency

please report it privately rather than opening a public issue. Use GitHub's private vulnerability reporting flow from the repository's **Security** tab. If private vulnerability reporting is temporarily unavailable, contact the maintainer through the GitHub profile listed in the README and do not include exploit details in a public issue.

Include:

- A description of the issue
- The steps to reproduce
- The version (commit hash) you tested
- Any suggested mitigation, if you have one

I will acknowledge receipt within seven days. If the report is confirmed, I will work on a fix and credit you (with your consent) in the changelog and any subsequent advisory.

## Out of Scope

The following are not in scope for this policy:

- Vulnerabilities in upstream dependencies (TerraTorch, Lightning, PyTorch, TerraMind weights). Please report those to the respective project.
- Issues with the cloud providers (Modal, Cerebrium, HuggingFace) themselves.
- Speculative findings without a reproducible exploit.

## Disclosure

I follow a 90-day coordinated-disclosure window. If a fix is not feasible within that period, I will publish the advisory together with mitigation guidance regardless.
