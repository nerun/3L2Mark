# AGENTS.md

Guidance for coding agents working on `codebench.py`.

## What this is

`codebench.py` is a single-file, standard-library-only Python benchmark that compares local LLMs on programming tasks. It talks to an OpenAI-compatible endpoint served by llama.cpp (default `http://127.0.0.1:9931`), asks the model to solve 21 tasks, compiles and runs every answer against hidden tests, and writes a Markdown report with a 0-100 score.

It exists to tell apart very similar models (fine-tunes and distills of the same 9B base). Everything follows from that:

- **Grading is objective only.** A test passes or fails by program. No visual judgement, no LLM judge, no style scoring.
- **Partial credit matters.** Tasks have many independent tests so that near-identical models land on different scores.
- **Noise is the enemy.** Anything that makes the same answer score differently on two runs or two machines is a bug.

## Commands

```bash
python3 codebench.py --selftest            # validate toolchains and every task (no model needed, ~2 min)
python3 codebench.py --selftest --only sql # subset: comma-separated ids or substrings
python3 codebench.py --list                # task ids, languages, categories
python3 codebench.py                       # full run against the server
python3 codebench.py --runs 3              # mean of 3 answers per task, reports the standard deviation
python3 codebench.py --compare             # ranking of all stored results
python3 codebench.py --resume results/<model>/<timestamp>
```

Toolchains on Debian: `sudo apt install gcc g++ default-jdk-headless php-cli mono-mcs`. SQL uses Python's built-in `sqlite3` (needs SQLite >= 3.35).

**Run `--selftest` after every change. It must end with `Self-test PASSED.`** Test C# on both toolchains when touching C# tasks or the build code: `--csharp mcs` and `--csharp dotnet`.

## Hard constraints

- One file, Python standard library only, no install step. The user runs one script.
- Linux only (`resource`, process groups, `/proc/cpuinfo`).
- All program messages, comments, prompts and reports are in English.
- The script never sends sampling parameters unless asked (`--temperature`, `--top-p`, `--extra-json`). The user configures the model on the server; the benchmark only sends `seed` and `max_tokens`.
- Each task is an independent request with no history.

## File layout (in order)

1. **Core**: `Rng`, `norm_out`, toolchain detection, `build_program`, `run_case`, `Server` (streaming client), `extract_code`, the `TASKS` registry and `task()`.
2. **Write tasks**: one per language, each with a Python reference function (`ref_*`), test builders and a reference solution in the target language.
3. **Bugfix tasks**: `*_GIVEN` (buggy program), `*_FIXES` (list of `(old, new)` replacements), `ref_*`, tests.
4. **Optimize tasks**: `*_GIVEN` (slow program), `*_FAST` (reference), tests with timed tiers.
5. **SQL tasks**: helpers (`sql_statements`, `run_steps`, `grade_behaviour`, `grade_query`) and six tasks with custom `grade` functions.
6. **Main**: prompts, `grade_program`, `grade_answer`, `selftest`, `write_report`, `compare`, `main`.

## Task model

A task is a dict registered with `task(...)`:

| Key | Meaning |
|---|---|
| `id`, `title` | `id` prefix encodes language and kind (`java_bugfix_sensors`) |
| `lang` | `python`, `c`, `cpp`, `csharp`, `php`, `java`, `sql` |
| `category` | `write`, `bugfix`, `optimize`, `sql` |
| `spec` | Specification text shown to the model |
| `given` | Buggy or slow source shown to the model (bugfix, optimize, some SQL) |
| `fixes` | `(old, new)` pairs turning `given` into a correct program (self-test only) |
| `solution` | Reference answer, string or callable (self-test only) |
| `tests` | Callable returning a list of `T(...)` cases (program tasks) |
| `grade` | Callable `(answer_sql, scale) -> [R(...)]` (SQL tasks) |
| `raw_prompt` | `True` when `spec` is already the complete prompt (all SQL tasks) |

Program tasks are stdin/stdout programs. Expected outputs are computed at run time by a Python reference (`ref_*`), never stored. A test is `T(name, input, expected, limit, weight, timed)`; `input` may be `str` or `bytes`.

- Non-timed tests check correctness. For C/C++ they run on an ASan+UBSan build, so memory errors fail the test.
- Timed tests (`timed=True`) run on the `-O2` build with `limit * --time-scale + STARTUP_ALLOWANCE[lang]`.
- Task score = passed weight / total weight. Final score = plain mean over tasks x 100.

## Invariants (enforced by `--selftest` or by convention)

1. **Every reference scores 100%.** For bugfix tasks the reference is `apply_fixes(given, fixes)`; each anchor must occur exactly once.
2. **Every `given` program scores below 50%** (currently 9-29%). In bugfix tasks only a few "base" tests may pass on the buggy code; in optimize tasks the small correctness tests weigh 0.5 each so that the unchanged program gets about 25%.
3. **Time limits are far from both sides.** Reference solutions take well under 1 s against limits of 2-4 s; slow originals do not finish at 5x the limit. Never add a tier where a plausible solution sits near the limit: that turns CPU speed into score noise. Measure before choosing sizes.
4. **Determinism.** Generate test data only with `Rng` (a fixed LCG) or closed-form arithmetic, never `random`. Results must be identical on every Python version.
5. **Each bug has isolated tests.** A test should fail because of one bug where possible, so fixing k of n bugs yields proportional credit. When bugs unavoidably interact, order tests from independent to combined.
6. **Specs are complete and unambiguous.** If two reasonable readings of the spec give different outputs, either fix the spec or keep that case out of the tests.
7. **Large expected values without slow references.** Where a Python reference would take seconds (inversions for n = 2,000,000, BFS on 10^6 cells), build inputs whose answer has a closed form (`_inv_blocks`, `_maze_serpentine`).

## Pitfalls already hit

- **`RLIMIT_FSIZE` kills the .NET runtime** (SIGXFSZ at start-up). C# and Java run without it; `run_case` polls the output file size instead. Keep `DOTNET_EnableWriteXorExecute=0`.
- **`dotnet build` needs NuGet restore (network).** C# is compiled by calling Roslyn directly (`dotnet <sdk>/Roslyn/bincore/csc.dll`) with a hand-written `runtimeconfig.json`. Do not switch back to MSBuild.
- **The C# target is part of the prompt** (`lang_requirements`): .NET if an SDK exists, otherwise Mono `mcs` (C# 7). Scores from different C# toolchains are not comparable.
- **Rounding ties.** Averages printed with two decimals must never land exactly on `x.xx5`: C, Python, .NET and Mono round ties differently. `avg_is_tie()` rejects such generated tests.
- **Accidentally masked bugs.** The Python closure bug was hidden because a later loop reused the same variable name. After editing a `given` program, check with `--selftest` that its score did not rise.
- **Runtime-dependent behaviour.** Avoid bugs that differ between runtimes (`Dictionary.Remove` during enumeration throws on Mono but not on .NET Core 3+; culture-aware string comparison; PHP 7 vs 8 loose comparison of non-numeric strings).
- **Programs run with `LC_ALL=C.UTF-8`** so that number formatting does not depend on the user's locale.
- **SQLite and correlated subqueries.** The slow query in `sql_opt_events` is only slow enough at the current tier sizes; re-measure the unchanged query if you touch them.
- **Row comparison** normalizes integral floats to `int` (`TOTAL()` vs `SUM()`), so do not write tests that depend on that difference.

## Adding or changing a task

1. Write the spec, a Python `ref_*`, hand-made edge cases and `Rng`-based random cases.
2. Write a reference solution in the target language (and `given` + `fixes` for bugfix tasks).
3. Register with `task(...)`; keep the categories balanced across languages.
4. Run `--selftest --only <id>`, check the original's score, and inspect per-test results for the `given` program to confirm each bug or tier behaves as designed.
5. For timed tiers, measure the reference, a typical "good enough" solution, and the slow original with `--time-scale 5`.

**Changing tasks, tests, prompts or weights changes what a score means.** Bump `VERSION` whenever you do. The version is stored in `results.json` (`meta.version`) and shown in `report.md`; `--compare` lists it per model and warns when versions differ, and `--resume` refuses a run created by another version.

## Results format

`results/<model>/<timestamp>/` holds `results.json` (meta + per task-run records), `report.md`, and `answers/<task>_run<k>.md|.<ext>` (prompt, reasoning, answer, extracted code). `--compare` and `--resume` read `results.json`; keep it backward compatible or bump `VERSION`.

## Known gaps

- Difficulty has **not been calibrated against real models**. Validation so far used reference solutions and a mock server. After real runs, look for tasks where every model scores 0 or 100: they add no discrimination and should be made easier, harder, or replaced.
- Single-shot only: no repair round after a compile error.
- Model-written code runs with the user's privileges (temporary directory, time and memory limits, no sandbox).
- `--resume` trusts the current command-line options instead of the stored ones.

## Versioning

The benchmark `codebench.py` is in Beta. The user will provide some results to the agent so that it can improve the test.

The first version is `1.0 Beta 1`. Whenever the agent changes the script, it should increment the Beta number by 1, without changing the value "1.0". It is not necessary to notify the user about the increments.