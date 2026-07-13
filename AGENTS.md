# Local workspace instructions

- Never create files, environments, caches, model weights, or installation data under
  `/home/kkumar`; that filesystem has limited storage.
- Keep the Conda environment in `.conda/` and all caches and temporary files in `.cache/`
  within this repository.
- Before working on the project, run `source scripts/activate_local_env.sh`. This activates
  the project environment and redirects common package, model, compiler, and plotting caches.
- Never commit tokens or credentials. Read the Hugging Face token from `HF_TOKEN`.

# Goal of this project

The user is a researcher working on the EgoLongQA task. Explain things conversationally
without compromising the technical discussion. Do not make answers sound like pointers from
a slide deck.

## Data and compute

- The EgoLongQA validation videos are stored in `/scratch/inf0/user/kkumar/val`.
- Use `configs/egolongqa_dev140_video_ids_seed20260709.json` as the canonical reduced
  validation split for faster evaluations. It contains 140 deterministically selected videos.
- The corresponding reduced evaluation samples are in
  `configs/egolongqa_dev140_seed20260709.json`.
- Access the Slurm submission host with `ssh slurm`. Run `gpucheck` on that host to inspect
  current GPU availability before choosing resources or submitting GPU jobs.
- Prefix commit subjects with a conventional tag such as `feat:`, `fix:`, `docs:`, or
  `refactor:`. Feature commits for this project should start with `feat:` and include a
  meaningful commit body describing the implementation and verification.

## Code style: avoid defensive overhead

Write code the way an ML researcher would write it for a colleague to read: direct, readable,
and free of unnecessary guardrails.

- Do not wrap operations in `try`/`except` unless a specific failure mode is expected and
  there is a meaningful way to handle it. Do not use bare or broad exception handlers merely
  to be safe.
- Do not add type, shape, or `None` checks, assertions, or input validation for conditions
  that cannot occur given how the function is called in this codebase. Assume well-formed
  inputs unless the function is a public API boundary.
- Do not add fallback branches, default values, or just-in-case logic for scenarios outside
  the actual requirements.
- When making an assumption instead of adding a check, state it in a one-line comment, for
  example: `# Input is always a 2D tensor.`
- Prefer code that fails loudly and immediately with its natural exception and traceback over
  code that silently swallows or masks errors.
