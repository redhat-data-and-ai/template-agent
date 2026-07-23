# Contributing

Thanks for your interest in contributing to Template Agent.

## Getting started

```bash
git clone https://github.com/redhat-data-and-ai/template-agent.git
cd template-agent
make install     # creates venv, installs deps + pre-commit hooks
make test        # run unit tests
```

## Branch strategy

- **`main`** is the stable release branch.
- **`deep-agent`** is the active development branch. Target your PRs here.
- Use short-lived feature branches off `deep-agent`.

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated changelog generation via release-please.

```
feat: add new skill for data validation
fix: correct thread cleanup on disconnect
docs: update MCP configuration section
chore: bump ruff to 0.8.0
```

Prefix types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`.

Breaking changes: add `!` after the type (e.g., `feat!: redesign config format`) or include a `BREAKING CHANGE:` footer.

## Signing off commits (DCO)

This project uses the [Developer Certificate of Origin (DCO)](https://developercertificate.org/). Every commit must include a `Signed-off-by` line certifying you have the right to submit it.

Add the `-s` flag when committing:

```bash
git commit -s -m "feat: add new feature"
```

This appends `Signed-off-by: Your Name <your@email.com>` using your Git config. The DCO check will fail on any PR with unsigned commits.

To sign off all commits in an existing branch retroactively:

```bash
git rebase HEAD~N --signoff
```

Replace `N` with the number of commits on your branch.

## Pull requests

1. Branch from `deep-agent`.
2. Make your changes. Keep PRs focused on a single concern.
3. Sign off every commit with `git commit -s`.
4. Run checks locally before pushing:
   ```bash
   pre-commit run --all-files
   make test
   ```
5. Open a PR targeting `deep-agent`. Fill in the PR template.
6. CI must pass (tests, pre-commit, vulnerability scan, DCO).
7. A CODEOWNERS review is required before merge.

## Code style

- **Formatting and linting**: ruff (enforced via pre-commit).
- **Type checking**: mypy (enforced via pre-commit).
- **Security scanning**: bandit (enforced via pre-commit).
- **Docstrings**: pydocstyle (enforced via pre-commit).

All of these run automatically on `git commit` if you ran `make install`.

## Testing

- Unit tests go in `tests/unit/`.
- Skill evaluations go in `config/agent/skills/*/evals/`.
- Minimum coverage threshold: 81%.

```bash
make test           # unit tests
make test-cov       # with coverage report
make test-all       # unit + skill evals
```

## Security

- Do not commit secrets, credentials, or `.env` files.
- Report vulnerabilities privately via [GitHub Security Advisories](https://github.com/redhat-data-and-ai/template-agent/security/advisories/new).
- See [SECURITY.md](SECURITY.md) for the full policy.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
