# Rules (apply to every agent)

## Scope and code style
1. Plain functions. No classes except `nn.Module`. No abstraction layers, config systems, logging frameworks, CLI frameworks, type-hint gymnastics, or design patterns. Constants at the top of the file.
2. One-line comments only where the reason is not obvious. No long docstrings.
3. Create only the files listed in `work.md` section 2. No extra scripts, notebooks, tests, utils, or folders.
4. Do not refactor, rename, or "improve" anything outside your jobs.

## Ownership and communication
5. Edit only files you own (`work.md` section 2). Reading anything is fine.
6. Talk to other agents only through `comms/`. Your report is the only file in `comms/` you write.
7. The contract (`work.md` section 3) is frozen. To change it, write a `CONTRACT REQUEST` in your report, then continue with the current contract until the human answers.
8. Before each job: read the other two reports. After each job, milestone, blocker or request: update your report (status, time, what changed). Keep entries short.
9. Signals use the exact phrases in `work.md` section 5.

## Working discipline
10. Verify, never assume. A job is DONE only after you ran its gate in `jobs.md` and wrote the command and a one-line result in your report. If you could not run it, the job is not done.
11. Never invent numbers (accuracy, timings, sizes). Every number in a report or README comes from a command you ran or a file in `outputs/`.
12. On an error: read the full traceback, fix the root cause, retry. After 3 failed attempts on the same problem, stop, write it under Blockers with the traceback's last lines, and move to your next independent job.
13. Do not wait idle. If blocked on another agent, work on a stub or your next independent job (`work.md` section 4).
14. Do not ask the human anything unless it is listed as a human task or you are blocked after rule 12.
15. Seed 42 everywhere (`config.SEED`).

## Environment
16. One shared venv at `.venv`, Python 3.10+. Install only packages in `requirements.txt` plus the ones your job needs; list any extra package under "Requests" to C so it lands in `requirements.txt`.
17. No `git` commands (add, commit, push) unless you are C and the human has approved. Never commit `data/`.
18. Do not download or open the dataset in the git tree outside `data/`.

## Honesty
19. If a result is bad (low accuracy, Grad-CAM looks at the background, quantization hurt), report it plainly with a likely cause. Do not hide it or tune on the test set.
20. Test set is used once, at the end, for reporting. Never for thresholds, early stopping, or model selection.
21. Finish with an `Explain-me` section (5 to 8 bullets) in your report. The human must be able to explain every choice in an interview.
