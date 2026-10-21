# SeeFood: agent entry point

The human tells you your letter in the first message: **A**, **B** or **C**. If no letter is given, ask once, then stop until answered.

Read in this order, nothing else first:
1. `rules.md` (all rules, applies to you)
2. `jobs.md` (only your section: your ordered jobs and their gates)
3. `agents/agent_<letter>.md` (detailed spec for your jobs)
4. `skills.md` (the "Shared" section and your own section)
5. `work.md` section 3 (frozen contract)
6. `comms/A_report.md`, `comms/B_report.md`, `comms/C_report.md`

Then do your jobs in order. Update only `comms/<letter>_report.md`. Do not do another agent's job even if it looks easy.

Start prompt for the human to paste: `You are Agent <letter>. Follow AGENTS.md.`
