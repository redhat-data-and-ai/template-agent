## What

<!-- One or two sentences: what changed and why. Link the issue. -->

Fixes #

## How

<!-- Describe your approach. Call out non-obvious design decisions, trade-offs, or alternatives you considered. -->

## Testing

<!-- How did you verify this works? -->

- [ ] Unit tests added/updated
- [ ] Ran locally (`uv run pytest tests/unit -x`)
- [ ] Manual verification (describe below if applicable)

## Rollback

<!-- How would you revert this if it breaks production? "Revert the commit" is fine for most changes. -->

## Checklist

- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `ci:`, etc.)
- [ ] No secrets, credentials, or PII in the diff
- [ ] No breaking changes (or documented above with a migration path)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
