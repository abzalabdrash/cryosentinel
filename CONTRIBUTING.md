# Contributing to CryoSentinel

Thank you for considering a contribution. CryoSentinel is a small open-source project maintained by one person, so some of the formal process you may have seen at larger projects is intentionally lightweight here.

## What contributions are welcome

- **Reproducibility checks.** If you re-ran any of the headline numbers and got a different result, please open an issue with the command, the GPU you used, the package versions, and the metric you obtained. This is the most useful kind of contribution.
- **Out-of-domain validation.** The model has not been validated outside High Mountain Asia. If you run the released checkpoint on the Andes, Alps, Caucasus, Patagonia, or any other glaciated region, please share the per-chip diagnostics. We will add the results (with attribution) to `docs/EXTERNAL_VALIDATION.md`.
- **Test-set audit.** We manually audited the seven lowest-IoU Ile Alatau chips. Independent review of additional low-IoU chips, especially outside Ile Alatau, is welcome.
- **Bug reports.** Specific, reproducible bug reports are very welcome. Please include the full traceback, the environment (`pip freeze`), and the exact command.
- **Documentation improvements.** Typos, unclear sentences, broken links, missing context — small PRs are welcome.

## What is currently out of scope

- **Architectural rewrites.** The decoder, the loss, and the SWA / EMA / TTA stack are locked at v1.0.0. We will revisit them in v2 once we have evidence that the v1 architecture has a real ceiling we cannot push past with data and configuration.
- **New backbones.** TerraMind 1.0 Large is the v1 backbone. A swap to a larger backbone (TerraMind 1.0 Huge, when released) is on the v2 roadmap.
- **Closed-data contributions.** All training and evaluation must be reproducible from publicly available data sources. We do not accept commits that depend on data the contributor cannot redistribute.

## How to submit a change

1. Open an issue describing what you want to change and why. For non-trivial changes, please wait for a maintainer comment before writing code, so we can avoid duplicate work.
2. Fork the repository and create a feature branch off `main`.
3. Run the test suite and the linters before pushing:

   ```bash
   pip install -e ".[dev]"
   pytest tests/
   ruff check src/ scripts/
   mypy src/cryosentinel
   ```

4. Open a pull request. Reference the issue number. Describe the change, the motivation, and any new tests you added.
5. The PR template will ask you to confirm that your contribution is licensed under Apache 2.0 (the same licence as the rest of the project).

## Style

- Python 3.11+. Type hints where they help; we are not strict about full mypy coverage yet.
- 100-column line length.
- Format with `ruff format`. Lint with `ruff check`. Both are configured in `pyproject.toml`.
- Docstrings: Google style. One-line summary, blank line, body. Cite specific papers (with author, year, and DOI or arXiv ID) when documenting a non-obvious algorithmic choice.
- Commits: imperative present tense ("Fix block-split off-by-one in lat hashing"), one logical change per commit when reasonable. Reference the issue number in the commit message.

## Reporting security issues

Please do not file public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible-disclosure procedure.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Briefly: be civil, be specific, be willing to be wrong.
