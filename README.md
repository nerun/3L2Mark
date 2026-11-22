# 3L2Mark

3L2Mark is an abbreviation for "LLLMMark" or **Local LLM Benchmark**. One way to test LLMs running on users' home GPUs and compare them objectively across various programming tests.

This benchmark was created because I wanted to evaluate which variant of the Qwen 3.5 9B was the best:

| Model | Note |
| --- | --- |
| MiMo-V2.6-Distill-Qwen-9B | Trained by Xiaomi using proprietary data |
| Ornith-1.5-9B | Diversified proprietary training |
| OxCoder-9B | Trained by Claude Fable 5.1 and GLM 5.3 |
| PentaCoder-9B | Merges Qwopus Coder, MiMo, Ornith, OxCoder, and Qwen3.8 |
| Qwen3.5-9B | Original |
| Qwen3.8-9B-Distill | 3.5 trained by 3.8 2.4T A95B |
| Qwopus3.5-9B-v3 | Trained by Claude Opus 4.6 |
| Qwopus3.5-9B-Coder | Optimized for agentic coding, tool calling, and logical reasoning |
| Qwythos-9B-v2 | Uncensored 3.5 trained by Claude Mythos/Fable |
| Swift-1.5-Ornith-9B | Derived from Ornith, efficient reasoning, -33% thinking tokens |
| Triumvirate-9B | Merges Qwopus Coder, MiMo, and Ornith |

For that, I asked Claude Fable 5.2 to create it... Yes, I know. But what could be better than an AI to test its little brethren?

## Requirements

- Linux only (the script uses `resource` and `/proc`).
- Python 3, standard library only, nothing to install with `pip`. This script was tested with Python 3.14.
- [llama.cpp](https://llama.app/) v1.0.0 or later, serving the model through its OpenAI-compatible API.
- The toolchains below. A missing toolchain only skips the tasks of that language, and the report marks the run as incomplete.

## Install dependencies

Toolchains on Debian:

```bash
sudo apt update && sudo apt install -y default-jdk-headless dotnet10 gcc g++ mono-mcs php-cli
```

C and C++ answers are tested on an AddressSanitizer + UndefinedBehaviorSanitizer build. On Debian, `gcc` and `g++` already pull in `libasan` and `libubsan`; on other distributions install the sanitizer runtimes separately if your compiler package does not include them.

SQL uses Python's built-in `sqlite3` (needs SQLite >= 3.35).

## Usage

Start a [`llama-server`](https://llama.app/), then run:

```sh
python3 codebench.py --help
```

By default the script talks to `http://127.0.0.1:9931`, so start the server on port 9931 (`--port 9931`). To use another host or port, pass `--url`:

```sh
python3 codebench.py --url http://127.0.0.1:8080
```

If you're running more than one execution (`--runs`), and need to interrupt (<kbd>Ctrl</kbd>+<kbd>c</kbd>), when you resume later, you should also enter the total number of runs you were executing:

```sh
python3 codebench.py --runs 3 --resume results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS>
```

To add more runs to a finished battery (for example, you ran `--runs 5` and now want 10), do not start a new run: `--compare` only reads the most recent result folder of each model, so a new folder would replace the old one in the report instead of adding to it. Resume the old folder and raise `--runs` to the new total. Only the missing runs (6 to 10) are asked, with new seeds, and the report then evaluates all 10:

```sh
python3 codebench.py --runs 10 --resume results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS>
```

Use the same server setup (model, quantization, `-ctk`/`-ctv`, context) and the same `--max-tokens` as in the first runs, because `--resume` uses the options of the current command line.

To redo a single answer (for example, one slowed down by something on your machine), list the stored records of a run, remove the one you want by its number, and resume with the same options as the original run:

```sh
python3 codebench.py --records results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS>
python3 codebench.py --records results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS> --remove 77
python3 codebench.py --runs 5 --resume results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS>
```

When a new version changes only a few tasks, a run made by an older version does not have to be redone from scratch. `--migrate` copies it to a new folder (`<YYYYMMDD-HHMMSS>-betaN`) without the answers of the changed tasks, leaving the original untouched; resuming the copy asks only those again:

```sh
python3 codebench.py --migrate results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS>
python3 codebench.py --runs 5 --resume results/<MODEL FOLDER>/<YYYYMMDD-HHMMSS>-betaN
```

When finished, use:

```sh
python3 codebench.py --compare
```

All models tested within the `results/` folder will be compared and ranked. The comparison report will be located in `results/comparison.md`, together with a formatted, self-contained copy in `results/comparison.html`.

The report calculates the average and standard deviation of each run for each model tested. Therefore, to maintain consistency in your tests, run the same number of times (`--runs`) for each model you are testing.
