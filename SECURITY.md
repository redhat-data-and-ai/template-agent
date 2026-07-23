# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| latest  | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Use [GitHub's private vulnerability reporting](https://github.com/redhat-data-and-ai/template-agent/security/advisories/new) to submit the details.
3. Include steps to reproduce, impact assessment, and any suggested fix.

We aim to acknowledge reports within 48 hours and provide a fix or mitigation within 7 days for critical issues.

## Security Measures

This project uses the following automated security tooling:

- **Trivy** -- container image vulnerability scanning on every build
- **Bandit** -- Python SAST via pre-commit
- **CodeQL** -- GitHub's semantic code analysis on every PR and weekly
- **Dependabot** -- automated dependency updates for pip, Docker, and GitHub Actions
- **Dependency Review** -- blocks PRs that introduce dependencies with known high/critical CVEs
- **OpenSSF Scorecard** -- weekly supply chain security health assessment
- **Cosign** -- keyless image signing for provenance verification
- **SBOM** -- CycloneDX bill of materials generated with every image build
