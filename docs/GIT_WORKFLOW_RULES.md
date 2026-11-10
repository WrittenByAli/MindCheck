# MindCheck: Git Pushing & Pulling Rules for Group 29

**Team Members**: **Muhammad Ali** (`24I-3158`) & **Shahzad Ahmad** (`24I-3090`)  
**Purpose**: Prevent merge conflicts, avoid broken builds, protect commit history, and ensure seamless parallel development.

---

## 1. The 5 Non-Negotiable Golden Rules

1. **Rule 1: NEVER Push Directly to `main`**  
   - `main` is the production branch. It must always be green, stable, and ready to demonstrate.
   - All code enters `main` exclusively through Pull Requests reviewed by your partner.

2. **Rule 2: NEVER Force Push (`git push -f`) on Shared Branches**  
   - A force push overwrites remote history and can permanently delete your partner's commits.
   - Use force push only on your private, unmerged feature branches if you are rebasing your own commits.

3. **Rule 3: ALWAYS Pull (`git pull --rebase`) Before Starting Work**  
   - Before writing a single line of code, synchronize your local repository with GitHub so you are building on the latest shared progress.

4. **Rule 4: NEVER Push Without Passing Local Tests**  
   - Before pushing any branch to GitHub, run `pytest tests/`. Pushing code that breaks CI halts both team members.

---

## 2. Standard Daily Workflow (Step-by-Step)

```
[Start Work] ──> git pull origin main ──> git checkout -b feat/subtask-x-y-z
                                                    │
                                                    ▼
                                          [Write Code & Tests]
                                                    │
                                                    ▼
                                          pytest tests/ (MUST PASS)
                                                    │
                                                    ▼
                                          git commit -m "feat(...): ..."
                                                    │
                                                    ▼
                                          git pull --rebase origin main
                                                    │
                                                    ▼
                                          git push -u origin feat/...
                                                    │
                                                    ▼
                                          [Open PR & Partner Reviews]
                                                    │
                                                    ▼
                                          [Squash & Merge to main]
```

### Step 1: Start from the Latest `main`
```bash
git checkout main
git pull origin main
```

### Step 2: Create a Feature Branch for Your Subtask
Use the naming pattern: `feat/subtask-<id>-<short-description>`
```bash
# Example for Shahzad:
git checkout -b feat/subtask-1.3.1-bitboards

# Example for Ali:
git checkout -b feat/subtask-2.4.1-streamlit-shell
```

### Step 3: Make Incremental, Logical Commits
Do not wait until the entire task is done to make one giant commit. Commit working logical units:
```bash
git add src/rules/fact_extractor.py tests/test_fact_extractor.py
git commit -m "feat(rules): implement pawn attack bitboards (Subtask 1.3.1)"
```

### Step 4: Rebase onto `main` Before Pushing
If your partner merged something while you were coding, rebase to bring in their changes cleanly:
```bash
git fetch origin
git rebase origin/main
```

### Step 5: Push Your Branch & Open a PR
```bash
git push -u origin feat/subtask-1.3.1-bitboards
```
- Open a Pull Request on GitHub.
- Assign your partner as the **Reviewer**.
- Wait for GitHub Actions CI to turn green.
- Once approved, click **Squash and Merge** and delete the remote branch.

---

## 3. Pre-Push Verification Checklist

Run this 3-command checklist before pushing any branch:

```bash
# 1. Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# 2. Check for syntax and linting errors
flake8 src/ tests/

# 3. Verify all unit tests pass
pytest tests/ -v
```

If all tests pass, check what files are about to be pushed:
```bash
git status
```
> [!WARNING]
> Verify that no large `.pgn` files (>10MB), `.pt` model weights, `.venv` folders, or `.DS_Store` / `Thumbs.db` files are staged!

---

## 4. How to Handle Merge Conflicts Cleanly

Merge conflicts happen when both members edit the exact same line of the same file. Follow these steps to resolve them without panic:

```bash
# 1. While on your feature branch, pull with rebase:
git pull --rebase origin main

# 2. Git will pause and flag conflicting files (e.g. config.yaml):
# Open the file in VS Code; you will see:
# <<<<<<< HEAD (Partner's code)
# ...
# =======
# ... (Your code)
# >>>>>>> feat/subtask-...

# 3. Edit the file to keep the correct combined code, remove conflict markers.

# 4. Stage the resolved file:
git add config/config.yaml

# 5. Continue the rebase:
git rebase --continue

# 6. Push your branch (if already pushed once, use safe lease force):
git push --force-with-lease
```

---

## 5. Communication Rule for Shared Files

Certain files affect both members:
- `requirements.txt` (installing a new package)
- `config/config.yaml` (changing a threshold or path)
- `docs/04_WORK_DISTRIBUTION_AND_OWNERSHIP.md` (interface data schemas)

**Rule**: If you need to edit a shared file, **drop a quick message in your team chat** (e.g., *"Hey, I added `onnxruntime` to requirements.txt, please pull before running"*). This eliminates 95% of all merge conflicts before they happen.
