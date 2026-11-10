# Contributing to MindCheck & Git Workflow Rules

Welcome to the **MindCheck** developer guide for **Group 29** (**Shahzad Ahmad** & **Muhammad Ali**). This document outlines our collaborative git workflow, coding standards, branch naming conventions, pushing/pulling rules, and peer review protocols.

---

## 1. The 4 Non-Negotiable Golden Rules

1. **Rule 1: NEVER Push Directly to `main`**  
   - `main` is the production branch. It must always be green, stable, and ready to demonstrate.
   - All code enters `main` exclusively through Pull Requests reviewed and approved by your partner.

2. **Rule 2: NEVER Force Push (`git push -f`) on Shared Branches**  
   - A force push overwrites remote history and can permanently delete your partner's commits.
   - Use force push only on private, unmerged feature branches with `git push --force-with-lease` after rebasing.

3. **Rule 3: ALWAYS Pull (`git pull --rebase origin main`) Before Starting Work**  
   - Before writing a single line of code, synchronize your local repository with GitHub so you are building on the latest shared progress.

4. **Rule 4: NEVER Push Without Passing Local Tests**  
   - Before pushing any branch to GitHub, run `pytest tests/`. Pushing code that breaks CI halts both team members.

---

## 2. Daily Step-by-Step Subtask Implementation Flow

```
[Start Session] ──> git checkout main ──> git pull origin main
                                                │
                                                ▼
                             git checkout -b feat/subtask-x-y-z
                                                │
                                                ▼
                                    [Write Code & PyTests]
                                                │
                                                ▼
                                    pytest tests/ (MUST PASS)
                                                │
                                                ▼
                                    git commit -m "feat(...): ..."
                                                │
                                                ▼
                                    git fetch origin
                                    git rebase origin/main
                                                │
                                                ▼
                             git push -u origin feat/subtask-x-y-z
                                                │
                                                ▼
                                    [Open PR on GitHub]
                                                │
                                                ▼
                           [Partner Reviews + CI Turns Green]
                                                │
                                                ▼
                                     [Squash & Merge to main]
```

### Step 1: Ensure `main` is Up-to-Date
```bash
git checkout main
git pull origin main
```

### Step 2: Create a Dedicated Feature Branch
Use the naming format: `feat/subtask-<id>-<short-description>`:
```bash
# Example for Shahzad:
git checkout -b feat/subtask-1.3.1-bitboard-extractor

# Example for Ali:
git checkout -b feat/subtask-2.4.1-streamlit-shell
```

### Step 3: Implement Code & Unit Tests
Follow the exact specification in `tasks/phase_X_.../subtasks/SUBTASK_...md`.

### Step 4: Run Pre-Push Quality Checks
```bash
# 1. Format code automatically
black src/ tests/ scripts/
isort src/ tests/ scripts/

# 2. Check for syntax and linting errors
flake8 src/ tests/

# 3. Run automated tests (MUST pass 100%)
pytest tests/ -v
```

### Step 5: Check Staged Files (No Heavy Artifacts!)
```bash
git status
```
> [!WARNING]
> Verify that no large `.pgn` files (>10MB), `.pt` model weights, `.venv` folders, or OS files (`Thumbs.db`, `.DS_Store`) are staged!

### Step 6: Commit with Conventional Commit Messages
```bash
git add src/rules/fact_extractor.py tests/test_fact_extractor.py
git commit -m "feat(rules): implement 64-bit bitboard fact extractor (Subtask 1.3.1)"
```

### Step 7: Rebase onto `main` and Push
```bash
git fetch origin
git rebase origin/main
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

## 4. How to Handle Merge Conflicts Cleanly

If your partner merged a PR into `main` while you were working on your branch, update your branch cleanly using **rebase**:

```bash
# 1. Fetch latest changes from GitHub
git fetch origin

# 2. Rebase your current feature branch on top of latest main
git rebase origin/main

# 3. If Git pauses on a conflicting file (e.g. config.yaml):
# Open the file in VS Code / editor, choose the correct combined code, remove conflict markers.

# 4. Stage the resolved file
git add config/config.yaml

# 5. Continue the rebase
git rebase --continue

# 6. Push your branch (use --force-with-lease safely if previously pushed)
git push --force-with-lease
```

---

## 5. Shared File Communication Rule

Three files directly affect both **Muhammad Ali** and **Shahzad Ahmad**:
1. `requirements.txt` (adding/updating dependencies)
2. `config/config.yaml` (modifying hyperparameters or paths)
3. `docs/04_WORK_DISTRIBUTION_AND_OWNERSHIP.md` (interface data contracts)

**Rule**: Before editing any of these shared files, send a brief message to your partner (e.g., *"Adding onnxruntime to requirements.txt, please pull before running"*). This simple habit eliminates 95% of all merge conflicts before they happen.

---

## 6. Peer Review & Pull Request Protocol

- **Assignee**: The member who implemented the subtask.
- **Reviewer**: The other group member:
  - If **Shahzad Ahmad** writes a PR, **Muhammad Ali** reviews it.
  - If **Muhammad Ali** writes a PR, **Shahzad Ahmad** reviews it.
- **Merge Requirements**:
  1. All GitHub Actions CI checks (`Code Quality & Linting`, `Unit & Integration Tests`, `Streamlit Headless`) must pass (green checkmark).
  2. The reviewer approves the PR after verifying adherence to the subtask specification.
  3. Use **Squash and Merge** or **Rebase and Merge** to keep git history clean.
