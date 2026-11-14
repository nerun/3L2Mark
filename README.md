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
| Triumvirate | Merges Qwopus Coder, MiMo, and Ornith |

For that, I asked Claude Fable 5.2 to create it... Yes, I know. But what could be better than an AI to test its little brethren?

## Install dependencies

Toolchains on Debian:

```bash
sudo apt update && sudo apt install -y default-jdk-headless dotnet10 gcc g++ mono-mcs php-cli
```

SQL uses Python's built-in `sqlite3` (needs SQLite >= 3.35).
