# 3L2Mark

3L2Mark is an abbreviation for "LLLMMark" or **Local LLM Benchmark**. One way to test LLMs running on users' home GPUs and compare them objectively across various programming tests.

This benchmark was created because I wanted to evaluate which variant of the Qwen 3.5 9B was the best:

- MiMo-V2.6-Distill-Qwen-9B
- Ornith-1.5-9B
- OxCoder-9B
- PentaCoder-9B
- Qwen3.5-9B (itself)
- Qwen3.8-9B-Distill
- Qwopus3.5-9B-v3
- Qwythos-9B-v2
- Triumvirate

For that, I asked Claude Fable 5.2 to create it... Yes, I know. But what could be better than an AI to test its little brothers?

## Install dependencies

Toolchains on Debian:

```bash
sudo apt update && sudo apt install -y default-jdk-headless dotnet10 gcc g++ mono-mcs php-cli
```

SQL uses Python's built-in `sqlite3` (needs SQLite >= 3.35).
