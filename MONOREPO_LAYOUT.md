# Monorepo Bootstrap: Hexagonal Kernel Layout

src/
  core/        # execution|risk|strategy
  adapters/    # exchanges|data
  infra/       # kafka|redis|db
  ops/         # observability|security
  cli/         # entrypoints

docs/
infra/
ops/
tests/

---

- All new code and refactors must follow this structure.
- Place dead code and legacy modules in /_archive or delete.
- See README.md for contribution and code quality rules.
