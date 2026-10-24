# Ronak's Profile

You are pair-programming with Ronak. Apply these preferences on top of the default profile.

## Environment (Critical)
- **Always run this first** in any new session: `export PATH="/home/ub/data/miniconda3/envs/monai/bin:$PATH"`
- Ronak's conda env: `monai` at `/home/ub/data/miniconda3/envs/monai` (Python 3.10)
- Use GPU 0 by default (`CUDA_VISIBLE_DEVICES=0`) unless told otherwise.
- Do NOT install `monai[all]` — it pulls old transformers requiring Rust compiler.

## Coding Preferences
- Prefer concise, direct explanations — skip unnecessary preambles.
- When writing Python, use type hints and docstrings.
- For deep learning projects, always use mixed precision (AMP) by default.
- Show training commands with `tmux` for long-running jobs.

## Project Context
- Primary project: BraTS Swin UNETR brain tumor segmentation at `/home/ub/ronak/dl/`
- Data lives at `/home/ub/ronak/dl/data/` (symlinked from NAS)
- MONAI 1.6 quirk: always use `Compose(lazy=False)` — lazy=True is the default and breaks spatial transforms.

## Git Protocol
- **NEVER stage files or make git commits automatically.** Do not run `git add` or `git commit`. Leave all git staging, commits, and pushes entirely to Ronak.

## Communication
- Be direct. If something is wrong, say so.
- Don't ask for confirmation on trivial actions (creating files, running safe commands).

