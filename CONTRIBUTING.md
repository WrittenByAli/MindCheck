# Contributing to MindCheck

Welcome to the **MindCheck** developer guide for **Group 29** (**Shahzad Ahmad** & **Muhammad Ali**). This document outlines our collaborative git workflow, coding standards, branch naming conventions, and testing protocols.

---

## 1. Git Workflow & Branching Strategy

To maintain an error-free, production-grade codebase:
1. **Never commit directly to `main`**. The `main` branch is always stable, deployable, and passing all CI checks.
2. **Branch per Subtask**: Create a dedicated feature branch for each subtask you implement:
   - Branch naming format: `feat/subtask-<id>-<short-description>`
   - Examples:
     - `feat/subtask-1.3.1-bitboards`
     - `feat/subtask-1.7.1-resnet-backbone`
     - `feat/subtask-2.4.1-streamlit-shell`
   - Bug fix branches: `fix/<issue-description>` (e.g., `fix/tensor-padding-mask`)

---

## 2. Step-by-Step Subtask Implementation Flow

When starting a subtask:
```bash
# 1. Ensure main branch is up to date
git checkout main
git pull origin main

# 2. Create your feature branch
git checkout -b feat/subtask-1.3.1-bitboard-extractor

# 3. Implement code & corresponding unit tests in tests/
# (Follow the exact specification in tasks/phase_X_.../subtasks/SUBTASK_...md)

# 4. Format and lint your code
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/

# 5. Run automated tests
pytest tests/ -v

# 6. Commit your changes using conventional commit messages
git add .
git commit -m "feat(rules): implement 64-bit bitboard fact extractor (Subtask 1.3.1)"

# 7. Push feature branch and open a Pull Request
git push -u origin feat/subtask-1.3.1-bitboard-extractor
```

---

## 3. Commit Message Standards

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat(...)`: A new feature or subtask implementation (e.g., `feat(models): implement residual CNN backbone`)
- `fix(...)`: A bug fix or tensor dimension correction (e.g., `fix(data): fix black perspective row flipping`)
- `test(...)`: Adding or updating test cases (e.g., `test(rules): add unit tests for pawn chain detection`)
- `docs(...)`: Documentation or report updates (e.g., `docs: update evaluation protocol`)
- `refactor(...)`: Code refactoring without changing functionality
- `ci(...)`: GitHub Actions workflow changes

---

## 4. Peer Review & Pull Request Protocol

- **Assignee**: The member who implemented the subtask.
- **Reviewer**: The other group member:
  - If **Shahzad Ahmad** writes a PR, **Muhammad Ali** reviews it.
  - If **Muhammad Ali** writes a PR, **Shahzad Ahmad** reviews it.
- **Merge Requirements**:
  1. All GitHub Actions CI checks (`Code Quality & Linting`, `Unit & Integration Tests`, `Streamlit Headless`) must pass (green checkmark).
  2. The reviewer approves the PR after verifying adherence to the subtask specification.
  3. Use **Squash and Merge** or **Rebase and Merge** to keep git history clean.
