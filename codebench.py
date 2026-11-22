#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Daniel Dias Rodrigues
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
codebench.py - objective coding benchmark for local LLMs served by llama.cpp.

Single file, Python standard library only. It talks to an OpenAI-compatible
endpoint (default http://127.0.0.1:9931), asks the model to solve programming
tasks in Python, C, C++, C#, PHP, Java and SQL, then compiles and runs every
answer against hidden test cases. Nothing is judged by eye: the grade (0-100)
is the fraction of hidden tests passed.

Typical use:
    python3 codebench.py                 # run the whole battery
    python3 codebench.py --selftest      # check toolchains and the battery itself (no model needed)
    python3 codebench.py --compare       # rank all models already tested

WARNING: this script compiles and executes code written by the model on your
machine (inside a temporary directory, with time and memory limits, but with
your user privileges).
"""

import argparse
import ctypes
import datetime
import http.client
import json
import math
import os
import re
import resource
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zlib
from fractions import Fraction
from html import escape as html_escape

VERSION = "1.0 Beta 20"
# Compatibility with codebench.py Beta engines: Beta numbers whose tasks, tests, prompts and
# weights are the same as in this version, so their scores can be mixed and compared.
COMPAT = [18, 19, 20]
# --migrate: runs made by these Beta versions can be brought to this version, because only the
# tasks in MIGRATE_REDO changed since then. Their records are dropped and --resume asks them again.
MIGRATE_FROM = [13, 14, 15, 16, 17]
MIGRATE_REDO = ["py_opt_subsum", "cs_write_sheet"]
DEFAULT_URL = "http://127.0.0.1:9931"
M64 = (1 << 64) - 1

# Extra seconds granted on top of every time limit (process start-up cost).
STARTUP_ALLOWANCE = {"python": 0.3, "c": 0.1, "cpp": 0.1, "csharp": 0.8,
                     "php": 0.3, "java": 1.2}

LANG_LABEL = {"python": "Python", "c": "C", "cpp": "C++", "csharp": "C#",
              "php": "PHP", "java": "Java", "sql": "SQL"}
LANG_FENCES = {
    "python": ("python", "py", "python3"),
    "c": ("c", "h"),
    "cpp": ("cpp", "c++", "cc", "cxx"),
    "csharp": ("csharp", "cs", "c#"),
    "php": ("php",),
    "java": ("java",),
    "sql": ("sql", "sqlite", "sqlite3"),
}
# Group used in the per-language breakdown of the report.
LANG_GROUP = {"python": "Python", "c": "C/C++", "cpp": "C/C++", "csharp": "C#",
              "php": "PHP", "java": "Java", "sql": "SQL"}
CATEGORY_LABEL = {"write": "Code quality (write from spec)",
                  "bugfix": "Hidden bugs (find and fix)",
                  "optimize": "Optimization",
                  "sql": "SQL (design, fix, optimize)"}


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

class Rng:
    """Tiny deterministic generator (identical on every Python version)."""

    def __init__(self, seed):
        self.s = (seed * 2654435761 + 0x9E3779B97F4A7C15) & M64
        for _ in range(4):
            self._step()

    def _step(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & M64
        return self.s >> 33

    def randint(self, a, b):
        span = b - a + 1
        if span <= (1 << 30):
            return a + self._step() % span
        return a + ((self._step() << 31) | self._step()) % span

    def choice(self, seq):
        return seq[self._step() % len(seq)]

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = self._step() % (i + 1)
            lst[i], lst[j] = lst[j], lst[i]

    def chance(self, percent):
        return self._step() % 100 < percent


def norm_out(text):
    """Normalize program output: line endings, trailing blanks, trailing empty lines."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def shorten(text, limit=600):
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]"


def is_jvm_noise(line):
    """JVM chatter that is not part of a tool's real output: the JAVA_TOOL_OPTIONS notice and
    unified-logging lines such as '[0.001s][warning][os,container] ...' (seen in containers)."""
    return line.startswith("Picked up ") or re.match(r"\[\d+[.,]\d+s\]\[", line) is not None


# The JVM writes its own warnings to stdout by default, which would corrupt a program's answer.
JVM_QUIET = ["-Xlog:disable", "-Xlog:all=warning:stderr"]


def first_line(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        out = p.stdout.decode("utf-8", "replace").strip().splitlines()
        out = [ln for ln in out if not is_jvm_noise(ln)]
        return out[0].strip() if out else ""
    except Exception:
        return ""


def is_tty():
    return sys.stderr.isatty()


def status(msg):
    """Transient one-line status on stderr (only when attached to a terminal)."""
    if is_tty():
        sys.stderr.write("\r\033[K" + msg)
        sys.stderr.flush()


def clear_status():
    if is_tty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


# --------------------------------------------------------------------------
# Toolchains
# --------------------------------------------------------------------------

def child_env():
    env = dict(os.environ)
    env.update({
        "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
        "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0:allocator_may_return_null=1",
        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=0",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1",
        "DOTNET_gcServer": "0", "DOTNET_EnableWriteXorExecute": "0", "DOTNET_TieredPGO": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.pop("JAVA_TOOL_OPTIONS", None)
    return env


def _dotnet_layout():
    """Locate the Roslyn compiler and the runtime reference assemblies of a .NET SDK."""
    dotnet = shutil.which("dotnet")
    if not dotnet:
        return None
    try:
        sdks = subprocess.run([dotnet, "--list-sdks"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=60, env=child_env()).stdout.decode()
        rts = subprocess.run([dotnet, "--list-runtimes"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=60, env=child_env()).stdout.decode()
    except Exception:
        return None

    def vkey(v):
        return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", v)[:3])

    sdk_list = []
    for ln in sdks.splitlines():
        m = re.match(r"(\S+)\s+\[(.+)\]", ln.strip())
        if m:
            csc = os.path.join(m.group(2), m.group(1), "Roslyn", "bincore", "csc.dll")
            if os.path.isfile(csc):
                sdk_list.append((vkey(m.group(1)), m.group(1), csc))
    rt_list = []
    for ln in rts.splitlines():
        m = re.match(r"Microsoft\.NETCore\.App\s+(\S+)\s+\[(.+)\]", ln.strip())
        if m:
            d = os.path.join(m.group(2), m.group(1))
            if os.path.isdir(d):
                rt_list.append((vkey(m.group(1)), m.group(1), d))
    if not sdk_list or not rt_list:
        return None
    sdk = max(sdk_list)
    usable = [r for r in rt_list if r[0][0] <= sdk[0][0]] or rt_list
    rt = max(usable)
    refs = []
    for fn in sorted(os.listdir(rt[2])):
        if fn.endswith(".dll") and (fn.startswith("System.") or fn.startswith("Microsoft.")
                                    or fn in ("mscorlib.dll", "netstandard.dll")):
            refs.append(os.path.join(rt[2], fn))
    return {"dotnet": dotnet, "csc": sdk[2], "sdk": sdk[1], "runtime": rt[1],
            "major": rt[0][0], "refs": refs}


def detect_toolchains(csharp_pref="auto"):
    """Return {lang: {"ok": bool, "desc": str, ...}} for every language of the battery."""
    tc = {}
    tc["python"] = {"ok": True, "desc": "Python " + sys.version.split()[0]}

    for lang, exe in (("c", "gcc"), ("cpp", "g++")):
        path = shutil.which(exe)
        tc[lang] = {"ok": bool(path), "desc": first_line([exe, "--version"]) if path else "",
                    "exe": path, "san": False, "hint": "gcc g++"}
    # Sanitizer availability (ASan + UBSan make memory errors an objective failure).
    if tc["c"]["ok"]:
        with tempfile.TemporaryDirectory(prefix="cb_san_") as d:
            src = os.path.join(d, "t.c")
            with open(src, "w") as f:
                f.write("int main(void){return 0;}\n")
            try:
                p = subprocess.run([tc["c"]["exe"], "-fsanitize=address,undefined", src, "-o",
                                    os.path.join(d, "t")], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=60)
                ok = p.returncode == 0 and any(subprocess.run(
                    [os.path.join(d, "t")], env=child_env(), timeout=30, preexec_fn=disable_aslr,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                    for _ in range(SAN_RETRIES))
            except Exception:
                ok = False
        tc["c"]["san"] = ok
        tc["cpp"]["san"] = ok and tc["cpp"]["ok"]

    php = shutil.which("php")
    tc["php"] = {"ok": bool(php), "desc": first_line(["php", "-v"]) if php else "", "exe": php,
                 "hint": "php-cli"}

    javac, java = shutil.which("javac"), shutil.which("java")
    tc["java"] = {"ok": bool(javac and java),
                  "desc": first_line(["javac", "-version"]) if javac else "",
                  "hint": "default-jdk-headless"}

    cs = {"ok": False, "desc": "", "hint": "mono-mcs (or a .NET SDK)"}
    mcs, mono = shutil.which("mcs"), shutil.which("mono")
    layout = _dotnet_layout() if csharp_pref in ("auto", "dotnet") else None
    if layout and csharp_pref in ("auto", "dotnet"):
        cs.update(ok=True, kind="dotnet", layout=layout,
                  desc=".NET SDK %s (runtime %s)" % (layout["sdk"], layout["runtime"]),
                  target=".NET %d (modern C#; a single file with a static Main method or "
                         "top-level statements)" % layout["major"])
    elif mcs and mono and csharp_pref in ("auto", "mcs"):
        cs.update(ok=True, kind="mcs", desc=first_line(["mono", "--version"]),
                  target="Mono (compiled with mcs): use C# 7.0 or older syntax only, a class "
                         "with a static Main method, and only the standard System.* libraries")
    tc["csharp"] = cs

    v = sqlite3.sqlite_version_info
    tc["sql"] = {"ok": v >= (3, 35, 0), "desc": "SQLite " + sqlite3.sqlite_version,
                 "hint": "a Python built against SQLite >= 3.35"}
    return tc


class Built:
    def __init__(self, ok, log="", cmd=None, cmd_san=None, warnings=0):
        self.ok, self.log, self.cmd, self.cmd_san, self.warnings = ok, log, cmd, cmd_san, warnings


def _run_tool(cmd, cwd, timeout=180):
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=child_env())
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "compiler timed out"
    except Exception as e:  # pragma: no cover
        return 125, "could not run %s: %s" % (cmd[0], e)


def build_program(lang, code, workdir, tc):
    """Write the source to workdir and compile it. Returns a Built object."""
    t = tc[lang]
    if lang == "python":
        src = os.path.join(workdir, "prog.py")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)
        rc, log = _run_tool([sys.executable, "-c",
                             "import sys;compile(open(sys.argv[1],encoding='utf-8').read(),"
                             "'prog.py','exec')", src], workdir)
        return Built(rc == 0, log, [sys.executable, "prog.py"])

    if lang in ("c", "cpp"):
        ext, std = ("c", "-std=gnu17") if lang == "c" else ("cpp", "-std=gnu++20")
        src = "prog." + ext
        with open(os.path.join(workdir, src), "w", encoding="utf-8") as f:
            f.write(code)
        rc, log = _run_tool([t["exe"], std, "-O2", "-Wall", "-Wextra", "-o", "prog", src, "-lm"],
                            workdir)
        if rc != 0:
            return Built(False, log)
        warnings = len(re.findall(r"\bwarning:", log))
        cmd_san = None
        if t.get("san"):
            rc2, _ = _run_tool([t["exe"], std, "-O1", "-g", "-fsanitize=address,undefined",
                                "-fno-sanitize-recover=all", "-fno-omit-frame-pointer",
                                "-o", "prog_san", src, "-lm"], workdir)
            if rc2 == 0:
                cmd_san = ["./prog_san"]
        return Built(True, log, ["./prog"], cmd_san, warnings)

    if lang == "csharp":
        with open(os.path.join(workdir, "Program.cs"), "w", encoding="utf-8") as f:
            f.write(code)
        if t["kind"] == "mcs":
            rc, log = _run_tool(["mcs", "-nologo", "-optimize+", "-out:prog.exe", "Program.cs"],
                                workdir)
            return Built(rc == 0, log, ["mono", "prog.exe"],
                         warnings=len(re.findall(r"\bwarning CS", log)))
        lay = t["layout"]
        with open(os.path.join(workdir, "GlobalUsings.cs"), "w") as f:
            f.write("global using System; global using System.Collections.Generic; "
                    "global using System.IO; global using System.Linq; "
                    "global using System.Threading; global using System.Threading.Tasks;\n")
        with open(os.path.join(workdir, "prog.runtimeconfig.json"), "w") as f:
            json.dump({"runtimeOptions": {
                "tfm": "net%d.0" % lay["major"],
                "framework": {"name": "Microsoft.NETCore.App", "version": lay["runtime"]},
                "configProperties": {"System.Globalization.Invariant": True}}}, f)
        with open(os.path.join(workdir, "refs.rsp"), "w") as f:
            f.write("\n".join('-r:"%s"' % r for r in lay["refs"]))
        rc, log = _run_tool([lay["dotnet"], lay["csc"], "-nologo", "-optimize+",
                             "-nullable:disable", "-nowarn:CS8632", "-out:prog.dll",
                             "@refs.rsp", "Program.cs", "GlobalUsings.cs"], workdir)
        return Built(rc == 0, log, [lay["dotnet"], "prog.dll"],
                     warnings=len(re.findall(r"\bwarning CS", log)))

    if lang == "php":
        if "<?php" not in code and "<?" not in code:
            code = "<?php\n" + code
        with open(os.path.join(workdir, "prog.php"), "w", encoding="utf-8") as f:
            f.write(code)
        rc, log = _run_tool([t["exe"], "-l", "prog.php"], workdir)
        return Built(rc == 0, log, [t["exe"], "-d", "display_errors=stderr",
                                    "-d", "memory_limit=1536M", "prog.php"])

    if lang == "java":
        m = re.search(r"^\s*public\s+(?:final\s+|abstract\s+)*class\s+([A-Za-z_]\w*)", code, re.M)
        main_class = m.group(1) if m else "Main"
        with open(os.path.join(workdir, main_class + ".java"), "w", encoding="utf-8") as f:
            f.write(code)
        rc, log = _run_tool(["javac", "-nowarn", "-encoding", "UTF-8", "-d", ".",
                             main_class + ".java"], workdir)
        log = "\n".join(ln for ln in log.splitlines() if not is_jvm_noise(ln))
        return Built(rc == 0, log, ["java"] + JVM_QUIET + ["-Xss256m", "-Xmx1500m", "-XX:+UseSerialGC",
                                                           "-cp", ".", main_class])
    raise ValueError(lang)


OUTPUT_CAP = 96 << 20

# Sanitizer builds can fail at start-up, at random, when address-space randomization puts a
# mapping where ASan wants its shadow memory (kernels with high mmap entropy, some containers).
# Two defences: start them with randomization off, and retry a crash that has no sanitizer report.
ADDR_NO_RANDOMIZE = 0x0040000
SAN_RETRIES = 5
try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
except Exception:
    _LIBC = None


def disable_aslr():
    """For preexec_fn: best effort, silently does nothing where the call is not permitted."""
    try:
        if _LIBC is not None:
            _LIBC.personality(ADDR_NO_RANDOMIZE)
    except Exception:
        pass


def has_sanitizer_report(err):
    return re.search(r"ERROR: (Address|Leak)Sanitizer|runtime error:|SUMMARY: \w+Sanitizer", err) is not None


def run_case(cmd, workdir, data, limit, mem_bytes=None, fsize_limit=True, no_aslr=False):
    """Run one test case. Returns (verdict, stdout_text, stderr_tail, seconds).

    verdict is one of: ok, timeout, crash.
    """
    in_path = os.path.join(workdir, "_in.txt")
    out_path = os.path.join(workdir, "_out.txt")
    err_path = os.path.join(workdir, "_err.txt")
    with open(in_path, "wb") as f:
        f.write(data)

    def limits():
        if no_aslr:
            disable_aslr()
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if fsize_limit:
            resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_CAP, OUTPUT_CAP))
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    t0 = time.monotonic()
    verdict = "ok"
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
        try:
            p = subprocess.Popen(cmd, cwd=workdir, stdin=fin, stdout=fout, stderr=ferr,
                                 env=child_env(), start_new_session=True, preexec_fn=limits)
        except Exception as e:
            return "crash", "", "could not start: %s" % e, 0.0
        rc = None
        note = ""
        while True:
            try:
                rc = p.wait(timeout=min(0.25, max(0.01, limit - (time.monotonic() - t0))))
                break
            except subprocess.TimeoutExpired:
                too_big = os.path.getsize(out_path) > OUTPUT_CAP
                if too_big or time.monotonic() - t0 >= limit:
                    verdict = "crash" if too_big else "timeout"
                    note = "output too large" if too_big else ""
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        p.kill()
                    p.wait()
                    break
        if verdict == "ok" and rc != 0:
            verdict = "crash"
            note = "killed by signal %d" % -rc if rc < 0 else "exit status %d" % rc
    secs = time.monotonic() - t0
    with open(out_path, "rb") as f:
        out = f.read(64 << 20).decode("utf-8", "replace")
    with open(err_path, "rb") as f:
        err = f.read(1 << 16).decode("utf-8", "replace")
    err = err[-1500:]
    if note:
        err = (err.rstrip() + "\n" + note).strip()
    return verdict, out, err, secs


# --------------------------------------------------------------------------
# llama.cpp client (OpenAI-compatible chat completions, streamed)
# --------------------------------------------------------------------------

CUT_OFF = ("length", "wall-limit", "loop")  # finish reasons of an answer that was not completed

LOOP_MIN_SPAN = 4000     # characters that must be an exact repetition
LOOP_MIN_REPS = 6        # ... and at least this many copies of the repeated block
LOOP_MAX_PERIOD = 2000   # longest repeated block looked for, in characters
LOOP_CHECK_EVERY = 1500  # characters received between two checks

# Second rule, for loops that change a little on every turn (a counter, a reworded line) and so
# are never an exact repetition: the last LOOP_Z_WINDOW characters compress to less than
# LOOP_Z_RATIO of their size on LOOP_Z_CHECKS checks in a row. Calibrated on 186 stored answers:
# answers that finished normally never stayed that repetitive for more than 2 checks in a row
# (a model can repeat itself for a while and still recover), the stuck ones did for 7 or more.
LOOP_Z_WINDOW = 4000
LOOP_Z_RATIO = 0.10
LOOP_Z_CHECKS = 6


def find_loop(text):
    """Return the period when the end of text is one block repeated verbatim, else 0.

    The rule is deliberately strict (thousands of characters of exact repetition), so that it
    only fires on a model stuck in a loop and never on ordinary repetitive code or prose.
    """
    n = len(text)
    for p in range(8, LOOP_MAX_PERIOD + 1):
        span = max(LOOP_MIN_SPAN, p * LOOP_MIN_REPS)
        if n < span + p:
            break
        if text[n - span:] == text[n - span - p:n - p]:
            return p
    return 0


class LoopWatch:
    """Loop detection over a stream (exact repetition, or lasting near-repetition); keeps only
    the tail it needs."""
    KEEP = LOOP_MAX_PERIOD * (LOOP_MIN_REPS + 1)

    def __init__(self):
        self.tail = ""
        self.pending = 0
        self.low = 0  # consecutive checks on which the tail was almost pure repetition

    def feed(self, piece):
        self.tail = (self.tail + piece)[-self.KEEP:]
        self.pending += len(piece)
        if self.pending < LOOP_CHECK_EVERY:
            return False
        self.pending = 0
        if find_loop(self.tail) > 0:
            return True
        if len(self.tail) >= LOOP_Z_WINDOW:
            raw = self.tail[-LOOP_Z_WINDOW:].encode("utf-8")
            self.low = self.low + 1 if len(zlib.compress(raw, 6)) < LOOP_Z_RATIO * len(raw) else 0
        return self.low >= LOOP_Z_CHECKS


class Server:
    def __init__(self, url, api_key=None):
        u = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        self.scheme = u.scheme or "http"
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or (443 if self.scheme == "https" else 80)
        self.prefix = u.path.rstrip("/")
        self.api_key = api_key
        self.url = "%s://%s:%d%s" % (self.scheme, self.host, self.port, self.prefix)
        self.model_id = None

    def _conn(self, timeout):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def _headers(self):
        h = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def get_json(self, path, timeout=15):
        try:
            c = self._conn(timeout)
            c.request("GET", self.prefix + path, headers=self._headers())
            r = c.getresponse()
            body = r.read()
            c.close()
            if r.status != 200:
                return None
            return json.loads(body.decode("utf-8", "replace"))
        except Exception:
            return None

    def probe(self):
        """Return a dict describing the server/model, or None when unreachable."""
        info = {}
        models = self.get_json("/v1/models")
        if models is None:
            health = self.get_json("/health")
            if health is None:
                return None
        else:
            try:
                self.model_id = models["data"][0]["id"]
                info["model_id"] = self.model_id
                meta = models["data"][0].get("meta") or {}
                for k in ("n_params", "size", "n_ctx_train"):
                    if k in meta:
                        info[k] = meta[k]
            except Exception:
                pass
        props = self.get_json("/props")
        if isinstance(props, dict):
            if props.get("model_path"):
                info["model_path"] = props["model_path"]
            if props.get("build_info"):
                info["build_info"] = props["build_info"]
            dgs = props.get("default_generation_settings") or {}
            if dgs.get("n_ctx"):
                info["n_ctx"] = dgs["n_ctx"]
            params = dgs.get("params") or dgs
            keep = {}
            for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty",
                      "presence_penalty", "frequency_penalty", "reasoning_format"):
                if k in params:
                    keep[k] = params[k]
            if keep:
                info["sampling"] = keep
        return info

    def chat(self, messages, max_tokens, seed, wall_limit, extra=None, idle_timeout=600,
             progress=None, detect_loops=True):
        """Streamed chat completion. Never raises; errors are reported in the result."""
        body = {"messages": messages, "stream": True, "max_tokens": max_tokens, "seed": seed,
                "stream_options": {"include_usage": True}}
        if self.model_id:
            body["model"] = self.model_id
        if extra:
            body.update(extra)
        res = {"content": "", "reasoning": "", "finish": None, "error": None,
               "completion_tokens": None, "prompt_tokens": None, "tps": None,
               "seconds": 0.0, "chunks": 0}
        t0 = time.monotonic()
        conn = None
        try:
            conn = self._conn(idle_timeout)
            conn.request("POST", self.prefix + "/v1/chat/completions",
                         body=json.dumps(body).encode("utf-8"), headers=self._headers())
            resp = conn.getresponse()
            if resp.status != 200:
                res["error"] = "HTTP %d: %s" % (resp.status,
                                                shorten(resp.read().decode("utf-8", "replace"), 300))
                return res
            ctype = resp.getheader("Content-Type", "")
            if "text/event-stream" not in ctype:
                # Server ignored "stream": parse a plain JSON answer.
                obj = json.loads(resp.read().decode("utf-8", "replace"))
                self._absorb_final(obj, res)
                return res
            content, reasoning = [], []
            tails = {"content": LoopWatch(), "reasoning": LoopWatch()}
            looping = False
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(obj.get("error"), (dict, str)):
                    res["error"] = shorten(json.dumps(obj["error"]), 300)
                    break
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        res["chunks"] += 1
                        looping = looping or (detect_loops and tails["content"].feed(delta["content"]))
                    r = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(r, str) and r:
                        reasoning.append(r)
                        res["chunks"] += 1
                        looping = looping or (detect_loops and tails["reasoning"].feed(r))
                    if ch.get("finish_reason"):
                        res["finish"] = ch["finish_reason"]
                self._absorb_usage(obj, res)
                elapsed = time.monotonic() - t0
                if progress:
                    progress(res["chunks"], elapsed)
                if elapsed > wall_limit and res["finish"] is None:
                    # A finished answer is not cut off: keep reading the usage chunk and [DONE].
                    res["finish"] = "wall-limit"
                    break
                if looping and res["finish"] is None:
                    res["finish"] = "loop"
                    break
            res["content"] = "".join(content)
            res["reasoning"] = "".join(reasoning)
            if res["finish"] is None and res["error"] is None:
                # The server went away in the middle of the answer: not the model's fault.
                res["error"] = "the stream ended without a finish reason (server stopped?)"
        except Exception as e:
            res["error"] = "%s: %s" % (type(e).__name__, e)
        finally:
            res["seconds"] = time.monotonic() - t0
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        if res["completion_tokens"] is None:
            res["completion_tokens"] = res["chunks"]
        if res["tps"] is None and res["seconds"] > 0 and res["completion_tokens"]:
            res["tps"] = res["completion_tokens"] / res["seconds"]
        return res

    @staticmethod
    def _absorb_usage(obj, res):
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if usage.get("completion_tokens") is not None:
                res["completion_tokens"] = usage["completion_tokens"]
            if usage.get("prompt_tokens") is not None:
                res["prompt_tokens"] = usage["prompt_tokens"]
        tm = obj.get("timings")
        if isinstance(tm, dict):
            if tm.get("predicted_n") is not None:
                res["completion_tokens"] = tm["predicted_n"]
            if tm.get("predicted_per_second"):
                res["tps"] = tm["predicted_per_second"]
            if tm.get("prompt_n") is not None and res["prompt_tokens"] is None:
                res["prompt_tokens"] = tm["prompt_n"]

    def _absorb_final(self, obj, res):
        try:
            ch = obj["choices"][0]
            msg = ch.get("message") or {}
            res["content"] = msg.get("content") or ""
            res["reasoning"] = msg.get("reasoning_content") or ""
            res["finish"] = ch.get("finish_reason")
        except Exception:
            res["error"] = "unexpected response: " + shorten(json.dumps(obj), 300)
        self._absorb_usage(obj, res)


# --------------------------------------------------------------------------
# Code extraction from a chat answer
# --------------------------------------------------------------------------

# A fence is a line that starts with three backticks; an inline ``` inside a sentence is not.
_FENCE_LINE = re.compile(r"[ \t]*```+[ \t]*([A-Za-z0-9_+#.\-]*)(.*)")
# A line of English prose: a capitalised word (not an all-caps SQL keyword) and three or more
# further words, without the characters that most code lines contain.
_PROSE_LINE = re.compile(r"(?:[-*]\s+|\d+[.)]\s+|\*\*)?[A-Z][a-z]\S*(?:\s+\S+){3,}")
_CODE_CHARS = re.compile(r"[;{}=]")


def _fence(line):
    """(tag, bare) if the line is a code fence, else None. A bare fence has no info string."""
    m = _FENCE_LINE.fullmatch(line)
    if not m:
        return None
    return m.group(1).lower(), not m.group(1) and not m.group(2).strip()


def _pair_fences(lines):
    """Split lines into fenced blocks. Returns (blocks, open_tail): blocks is a list of
    (tag, body, closed); open_tail is True when the last block runs to the end of the text.
    A fence with an info string inside an open block closes it and opens a new one: models
    never put "```python" inside a code block, but they do forget a closing fence."""
    blocks, tag, start = [], None, None
    for i, line in enumerate(lines):
        f = _fence(line)
        if f is None:
            continue
        if start is None:
            tag, start = f[0], i + 1
        elif f[1]:
            blocks.append((tag, "\n".join(lines[start:i]), True))
            start = None
        else:
            blocks.append((tag, "\n".join(lines[start:i]), True))
            tag, start = f[0], i + 1
    if start is not None:
        blocks.append((tag, "\n".join(lines[start:]), False))
    return blocks, start is not None


# A short introduction such as "Here is the program:".
_INTRO_LINE = re.compile(r"[A-Z][a-z][^;{}=()]*:")


def _is_prose(line):
    line = line.strip()
    return bool(_PROSE_LINE.fullmatch(line)) and not _CODE_CHARS.search(line)


def _code_likeness(text):
    """Lines that do not read as prose minus lines that do (blank lines do not count)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    prose = sum(1 for ln in lines if _is_prose(ln))
    return len(lines) - 2 * prose


def _drop_intro(lines):
    """Remove the prose lines that introduce code written without its opening fence."""
    i = 0
    while i < len(lines) and (not lines[i].strip() or _is_prose(lines[i])
                              or _INTRO_LINE.fullmatch(lines[i].strip())):
        i += 1
    return lines[i:]


def extract_code(answer, lang):
    """Pick the program out of a chat answer.

    Rule: among fenced blocks tagged with the expected language take the longest; if there
    is none, the longest fenced block; if there is no fence at all, the whole answer.
    A missing fence must not cost points, because a person reading the answer would not be
    stopped by it: a block left open runs to the end of the answer, and when the opening
    fence is missing (code first, then a lone closing fence) the text before that fence is
    taken as a block (without introductory prose lines), if it reads more like code than the
    text the fence would otherwise open.
    Returns (code, how).
    """
    text = answer
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks, open_tail = _pair_fences(lines)
    first = next((i for i, ln in enumerate(lines) if _fence(ln)), None)
    if open_tail and _fence(lines[first])[1]:
        # The first fence may be the closing fence of a block whose opening fence is missing.
        pre = "\n".join(_drop_intro(lines[:first]))
        if _code_likeness(pre) > _code_likeness(blocks[-1][1]):
            rest, _ = _pair_fences(lines[first + 1:])
            blocks = [("", pre, "stray")] + rest
    blocks = [b for b in blocks if b[1].strip()]
    if not blocks:
        return text.strip() + "\n", "no code fence (whole answer used)"
    tagged = [b for b in blocks if b[0] in LANG_FENCES[lang]]
    pool = tagged or blocks
    tag, code, closed = max(pool, key=lambda b: len(b[1]))
    how = "fenced block" if tagged else "fenced block (language tag missing or different)"
    if closed == "stray":
        how = "code before a lone closing fence (opening fence missing)"
    elif not closed:
        how += " (closing fence missing)"
    if len(pool) > 1:
        how += ", longest of %d" % len(pool)
    return code.strip("\n") + "\n", how


# --------------------------------------------------------------------------
# Task registry
# --------------------------------------------------------------------------

TASKS = []


def task(**kw):
    TASKS.append(kw)


RESPONSE_RULES = (
    "Answer with the complete source code of ONE program in a single fenced code block. "
    "The program must read from standard input and write to standard output, exactly in the "
    "formats described. Do not print prompts, debug text or anything else. "
    "Use only the language's standard library."
)


def lang_requirements(lang, tc):
    if lang == "python":
        return "Language: Python 3 (standard library only)."
    if lang == "c":
        return "Language: C (C17, compiled with gcc -O2). It must also run cleanly under AddressSanitizer and UndefinedBehaviorSanitizer."
    if lang == "cpp":
        return "Language: C++ (C++20, compiled with g++ -O2). It must also run cleanly under AddressSanitizer and UndefinedBehaviorSanitizer."
    if lang == "csharp":
        return "Language: C#. Target: %s." % tc["csharp"].get("target", "a recent C# compiler")
    if lang == "php":
        return "Language: PHP 8, run from the command line (php prog.php). Core functions only: no bcmath, gmp or mbstring. Start the file with <?php."
    if lang == "java":
        return "Language: Java 17 or newer, a single file whose public class is named Main (no package declaration)."
    return ""


def T(name, inp, exp, limit=10.0, weight=1.0, timed=False):
    """One hidden test case. timed=True: optimized binary, strict time limit."""
    return {"name": name, "input": inp, "expected": exp, "limit": limit, "weight": weight,
            "timed": timed}


# ==========================================================================
# WRITE-FROM-SPEC TASKS (one per language)
# ==========================================================================

# ---- Python: semantic version sorting and constraints ---------------------

SEMVER_SPEC = r"""Write a program that sorts semantic versions and resolves version constraints.

INPUT
Line 1: N (1 <= N <= 2000). Next N lines: one version each, written MAJOR.MINOR.PATCH, optionally followed by -PRERELEASE and/or +BUILD (examples: 1.4.0, 2.0.0-rc.1, 1.0.0-alpha.beta+exp.sha.5114f85). MAJOR, MINOR and PATCH are non-negative integers without leading zeros. PRERELEASE and BUILD are dot-separated identifiers made of the characters [0-9A-Za-z-].
Next line: Q (0 <= Q <= 2000). Next Q lines: one constraint each.

PRECEDENCE (Semantic Versioning 2.0.0)
- Compare MAJOR, MINOR and PATCH numerically, in that order.
- A version with a prerelease has LOWER precedence than the same MAJOR.MINOR.PATCH without one.
- Two prereleases are compared identifier by identifier from left to right: identifiers made only of digits are compared numerically; other identifiers are compared as ASCII strings; a numeric identifier is always lower than a non-numeric one; if all shared identifiers are equal, the prerelease with fewer identifiers is lower.
- BUILD metadata is ignored for precedence.

CONSTRAINTS
A constraint is one or more comparators separated by single spaces; a version satisfies it when it satisfies ALL comparators. A comparator is an operator immediately followed by a version (never with build metadata):
  =V  >V  >=V  <V  <=V    plain precedence comparisons (=V means "same precedence as V")
  ^X.Y.Z    >=X.Y.Z and <(X+1).0.0    (X is never 0 here)
  ~X.Y.Z    >=X.Y.Z and <X.(Y+1).0
The version after ^ or ~ never carries a prerelease.
Every comparison uses precedence only. There is NO special treatment of prereleases: for example 2.0.0-rc.1 DOES satisfy <2.0.0 and ^1.2.0.

OUTPUT
First the N versions in ascending precedence, one per line, printed exactly as given (including build metadata). Versions with equal precedence keep their relative input order.
Then, for each constraint, one line with the satisfying version that appears LAST in that sorted list, or the word none when no version satisfies it."""


def _sv_key(v):
    v = v.split("+", 1)[0]
    core, sep, pre = v.partition("-")
    a, b, c = (int(x) for x in core.split("."))
    if not sep:
        return (a, b, c, 1, ())
    ids = tuple((0, int(x), "") if x.isdigit() else (1, 0, x) for x in pre.split("."))
    return (a, b, c, 0, ids)


def _sv_bounds(comp):
    """Return a list of (op, key) for one comparator."""
    for op in (">=", "<=", ">", "<", "=", "^", "~"):
        if comp.startswith(op):
            k = _sv_key(comp[len(op):])
            if op == "^":
                return [(">=", k), ("<", (k[0] + 1, 0, 0, 1, ()))]
            if op == "~":
                return [(">=", k), ("<", (k[0], k[1] + 1, 0, 1, ()))]
            return [(op, k)]
    raise ValueError(comp)


def _sv_ok(key, op, ref):
    if op == "=":
        return key == ref
    if op == ">":
        return key > ref
    if op == ">=":
        return key >= ref
    if op == "<":
        return key < ref
    return key <= ref


def ref_semver(inp):
    lines = inp.split("\n")
    n = int(lines[0])
    versions = [ln.strip() for ln in lines[1:1 + n]]
    q = int(lines[1 + n])
    constraints = [ln.strip() for ln in lines[2 + n:2 + n + q]]
    ordered = sorted(versions, key=_sv_key)
    keys = [_sv_key(v) for v in ordered]
    out = list(ordered)
    for c in constraints:
        bounds = []
        for comp in c.split():
            bounds.extend(_sv_bounds(comp))
        found = "none"
        for i in range(len(ordered) - 1, -1, -1):
            if all(_sv_ok(keys[i], op, ref) for op, ref in bounds):
                found = ordered[i]
                break
        out.append(found)
    return "\n".join(out) + "\n"


def _semver_input(versions, constraints):
    return "%d\n%s\n%d\n%s" % (len(versions), "\n".join(versions), len(constraints),
                               "".join(c + "\n" for c in constraints))


def _semver_random(seed, n, q):
    r = Rng(seed)
    pres = ["alpha", "beta", "rc", "rc-1", "0", "1", "2", "10", "11", "x", "X", "0a", "dev"]
    builds = ["build", "1", "sha.5114f85", "exp", "20260101"]

    def version(with_build=True, plain=False, min_major=0):
        s = "%d.%d.%d" % (r.randint(min_major, 3), r.randint(0, 12), r.randint(0, 12))
        if not plain and r.chance(45):
            s += "-" + ".".join(r.choice(pres) for _ in range(r.randint(1, 3)))
        if with_build and not plain and r.chance(25):
            s += "+" + r.choice(builds)
        return s

    versions = [version() for _ in range(n)]
    ops = [">=", "<=", ">", "<", "=", "^", "~"]
    cons = []
    for _ in range(q):
        parts = []
        for _ in range(r.randint(1, 2)):
            op = r.choice(ops)
            if op == "^":
                parts.append(op + version(plain=True, min_major=1))
            elif op == "~":
                parts.append(op + version(plain=True))
            else:
                parts.append(op + version(False))
        if r.chance(15):
            parts = ["=" + r.choice(versions).split("+")[0]]
        cons.append(" ".join(parts))
    return _semver_input(versions, cons)


def semver_tests():
    # The first tests have no constraints at all, and the constraint tests use plain versions,
    # so that a program with one broken feature still earns the others.
    plain = ["1.0.0", "2.1.0", "1.10.0", "1.9.0", "0.3.7", "1.2.10", "1.2.9", "10.0.0", "0.0.9"]
    cases = [
        ("sort: plain versions", _semver_input(plain, [])),
        ("sort: prereleases", _semver_input(
            ["1.0.0-rc.1", "1.0.0-beta.11", "1.0.0", "1.0.0-alpha.beta", "1.0.0-alpha",
             "1.0.0-beta.2", "1.0.0-alpha.1", "1.0.0-beta", "0.9.9", "1.0.1-alpha"], [])),
        ("sort: hyphens and mixed identifiers", _semver_input(
            ["1.0.0-alpha-1", "1.0.0-alpha", "1.0.0-0alpha", "1.0.0-10", "1.0.0-9",
             "1.0.0-a.b.c", "1.0.0-a.b", "1.0.0-A", "1.0.0-rc-1.2", "1.0.0-rc-1.10",
             "1.0.0--", "1.0.0-a.1", "1.0.0-a.a"], [])),
        ("sort: build metadata and stability", _semver_input(
            ["1.0.0+b", "1.0.0-x+z", "1.0.0+a", "1.0.0", "1.0.0-x+a", "1.0.0-x", "0.9.0+zzz"], [])),
        ("constraints: one comparator", _semver_input(
            plain, [">=1.0.0", "<1.0.0", "=2.1.0", ">2.1.0", "<=0.3.7", ">1.2.9", "=3.0.0", "<0.0.9"])),
        ("constraints: several comparators and none", _semver_input(
            plain, [">1.2.9 <1.9.0", ">=1.0.0 <2.0.0", ">5.0.0 <9.0.0", ">=0.3.7 <=0.3.7",
                    ">1.10.0 <2.1.0", ">=1.2.9 <1.2.10 >0.0.1"])),
        ("constraints: caret and tilde", _semver_input(
            ["0.2.5", "0.3.0", "1.2.3", "1.9.9", "2.0.0", "1.2.10", "1.3.0", "0.2.1", "3.4.5"],
            ["^1.2.3", "~1.2.3", "~0.2.0", "^2.0.1", "~1.9.0 <1.9.9", "^1.0.0 <1.3.0", "~3.4.6",
             "^3.0.0"])),
        ("constraints: prereleases and build metadata", _semver_input(
            ["1.0.0-rc.1", "1.0.0-beta.11", "1.0.0+b", "1.0.0-alpha", "1.0.0-beta.2", "1.0.0+a",
             "2.0.0-rc.1", "1.2.3"],
            ["<1.0.0", "<1.0.0-beta.3", ">=1.0.0-beta.3 <1.0.0-rc.1", "=1.0.0", "=1.0.0-alpha",
             ">1.0.0-rc.1 <1.2.3", "<2.0.0", "^1.2.0", "<1.0.0-alpha"])),
    ]
    tests = [T(name, inp, ref_semver(inp), limit=10.0) for name, inp in cases]
    for i, (n, q) in enumerate([(30, 30), (400, 300), (2000, 2000)]):
        inp = _semver_random(101 + i, n, q)
        tests.append(T("random %d versions" % n, inp, ref_semver(inp), limit=10.0))
    return tests


def semver_solution():
    import inspect
    src = "import sys\n\n"
    for fn in (_sv_key, _sv_bounds, _sv_ok, ref_semver):
        src += inspect.getsource(fn) + "\n\n"
    return src + "sys.stdout.write(ref_semver(sys.stdin.read()))\n"


task(id="py_write_semver", lang="python", category="write",
     title="Semantic version sort and constraint resolver",
     spec=SEMVER_SPEC, tests=semver_tests, solution=semver_solution)


# ---- C++: limit order book -------------------------------------------------
# The reference and the solution also implement a CANCEL command that the task no longer has.

ORDERBOOK_SPEC = r"""Write a limit order book matching engine.

INPUT
Up to 200000 lines, read until end of input. Each line is one command:
  BUY id price qty
  SELL id price qty
  TOP
id, price and qty are positive integers (id and price <= 10^9, qty <= 10^6). Every id is used by one line only.

RULES
- The book holds resting orders. A new BUY order matches resting SELL orders whose price is <= the buy price; a new SELL order matches resting BUY orders whose price is >= the sell price.
- Matching priority: best price first (lowest price among resting SELLs, highest price among resting BUYs); among resting orders with the same price, the one that arrived earlier goes first.
- Each match trades the minimum of the two remaining quantities at the RESTING order's price and prints:  TRADE buy_id sell_id price qty
- Matching continues until the incoming order is completely filled or no resting order qualifies; any remaining quantity then rests in the book.
- TOP prints  TOP bid_price bid_qty ask_price ask_qty  where bid_price is the highest resting BUY price and bid_qty is the total remaining quantity of all resting BUY orders at that price; ask_price/ask_qty are the same for the lowest resting SELL price. When a side is empty print - - for its two fields (example: TOP 100 30 - -).

OUTPUT
The lines described above, in the order the events happen. Quantity totals can exceed 32 bits.
Performance matters: 200000 commands on a deep book must be processed in well under one second."""


def ref_orderbook(inp):
    import heapq
    out = []
    heaps = {"B": [], "S": []}
    live = {}
    level = {"B": {}, "S": {}}
    seq = 0

    def best(side):
        h = heaps[side]
        while h and h[0][2] not in live:
            heapq.heappop(h)
        return h[0] if h else None

    for line in inp.split("\n"):
        p = line.split()
        if not p:
            continue
        if p[0] in ("BUY", "SELL"):
            oid, price, qty = int(p[1]), int(p[2]), int(p[3])
            seq += 1
            side = "B" if p[0] == "BUY" else "S"
            opp = "S" if side == "B" else "B"
            while qty > 0:
                top = best(opp)
                if top is None:
                    break
                rid = top[2]
                r = live[rid]
                if (side == "B" and r[1] > price) or (side == "S" and r[1] < price):
                    break
                f = min(qty, r[2])
                if side == "B":
                    out.append("TRADE %d %d %d %d" % (oid, rid, r[1], f))
                else:
                    out.append("TRADE %d %d %d %d" % (rid, oid, r[1], f))
                qty -= f
                r[2] -= f
                level[opp][r[1]] -= f
                if r[2] == 0:
                    del live[rid]
            if qty > 0:
                live[oid] = [side, price, qty]
                heapq.heappush(heaps[side], (-price if side == "B" else price, seq, oid))
                level[side][price] = level[side].get(price, 0) + qty
        elif p[0] == "CANCEL":
            oid = int(p[1])
            if oid in live:
                r = live.pop(oid)
                level[r[0]][r[1]] -= r[2]
                out.append("CANCELLED %d" % oid)
            else:
                out.append("NOT_FOUND %d" % oid)
        elif p[0] == "TOP":
            b, s = best("B"), best("S")
            bs = "%d %d" % (-b[0], level["B"][-b[0]]) if b else "- -"
            ss = "%d %d" % (s[0], level["S"][s[0]]) if s else "- -"
            out.append("TOP %s %s" % (bs, ss))
    return "\n".join(out) + ("\n" if out else "")


def _orderbook_random(seed, n, spread, deep=False):
    r = Rng(seed)
    lines = []
    ids = []
    next_id = 1
    mid = 50000
    for i in range(n):
        x = r.randint(1, 100)
        if deep and i < n // 2:
            # Build a deep, non-crossing book first.
            side = "BUY" if r.chance(50) else "SELL"
            price = mid - r.randint(1, spread) if side == "BUY" else mid + r.randint(1, spread)
            lines.append("%s %d %d %d" % (side, next_id, price, r.randint(1, 1000)))
            ids.append(next_id)
            next_id += 1
        elif x <= 62:
            side = "BUY" if r.chance(50) else "SELL"
            off = r.randint(-spread // 4, spread)
            price = mid - off if side == "BUY" else mid + off
            if deep:
                price = mid + r.randint(-40, 40)
            lines.append("%s %d %d %d" % (side, next_id, max(1, price), r.randint(1, 1000)))
            ids.append(next_id)
            next_id += 1
        else:
            lines.append("TOP")
    lines.append("TOP")
    return "\n".join(lines) + "\n"


def orderbook_tests():
    cases = [
        ("one full match", "SELL 1 100 10\nBUY 2 100 10\n"),
        ("no match: orders rest", "SELL 1 100 10\nBUY 2 99 4\nTOP\nSELL 3 101 1\nBUY 4 98 1\nTOP\n"),
        ("partial fills", "SELL 1 100 10\nBUY 2 100 4\nTOP\nBUY 3 100 4\nTOP\nBUY 4 100 4\nTOP\n"
                          "SELL 5 100 1\nSELL 6 100 5\nTOP\n"),
        ("trade at the resting price", "SELL 1 100 5\nBUY 2 110 5\nBUY 3 90 5\nSELL 4 80 5\n"
                                       "BUY 5 50 2\nSELL 6 40 1\nTOP\n"),
        ("price priority",
         "SELL 1 105 5\nSELL 2 101 5\nSELL 3 103 5\nBUY 4 104 12\nTOP\nBUY 5 90 7\nBUY 6 95 7\n"
         "BUY 7 92 7\nSELL 8 91 16\nTOP\n"),
        ("time priority at one price",
         "BUY 10 50 3\nBUY 11 50 4\nBUY 12 50 5\nTOP\nSELL 13 50 5\nTOP\nSELL 14 40 100\nTOP\n"
         "BUY 15 45 1\nTOP\n"),
        ("top aggregation and empty sides",
         "TOP\nBUY 1 7 1000000\nBUY 2 7 1000000\nBUY 3 6 5\nTOP\nSELL 4 9 2\nSELL 5 9 3\n"
         "SELL 6 8 1\nTOP\nBUY 7 8 1\nTOP\nSELL 8 7 2000000\nTOP\nSELL 9 1 5\nTOP\n"),
        ("large quantities", "".join("BUY %d 500 1000000\n" % i for i in range(1, 3001))
         + "TOP\nSELL 9000 1 999999\nSELL 9001 500 2500000000\nTOP\n"),
    ]
    tests = [T(name, inp, ref_orderbook(inp)) for name, inp in cases]
    for i, (n, spread) in enumerate([(200, 10), (5000, 60)]):
        inp = _orderbook_random(201 + i, n, spread)
        tests.append(T("random %d commands" % n, inp, ref_orderbook(inp)))
    inp = _orderbook_random(299, 200000, 45000, deep=True)
    tests.append(T("performance: 200000 commands, deep book", inp, ref_orderbook(inp),
                   limit=2.0, weight=2.0, timed=True))
    return tests


ORDERBOOK_SOLUTION = r'''#include <bits/stdc++.h>
using namespace std;
struct Order { int side; long long price; long long qty; list<int>::iterator it; };
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    map<long long, list<int>> book[2];
    map<long long, long long> lvl[2];
    unordered_map<int, Order> live;
    string cmd, out;
    while (cin >> cmd) {
        if (cmd == "BUY" || cmd == "SELL") {
            int id; long long price, qty;
            cin >> id >> price >> qty;
            int side = cmd == "BUY" ? 0 : 1, opp = 1 - side;
            while (qty > 0 && !book[opp].empty()) {
                auto lv = (opp == 1) ? book[opp].begin() : prev(book[opp].end());
                long long rp = lv->first;
                if ((side == 0 && rp > price) || (side == 1 && rp < price)) break;
                int rid = lv->second.front();
                Order &r = live[rid];
                long long f = min(qty, r.qty);
                out += "TRADE " + to_string(side == 0 ? id : rid) + " " + to_string(side == 0 ? rid : id)
                     + " " + to_string(rp) + " " + to_string(f) + "\n";
                qty -= f; r.qty -= f; lvl[opp][rp] -= f;
                if (r.qty == 0) {
                    lv->second.pop_front();
                    live.erase(rid);
                    if (lv->second.empty()) { lvl[opp].erase(rp); book[opp].erase(lv); }
                }
            }
            if (qty > 0) {
                auto &l = book[side][price];
                l.push_back(id);
                live[id] = Order{side, price, qty, prev(l.end())};
                lvl[side][price] += qty;
            }
        } else if (cmd == "CANCEL") {
            int id; cin >> id;
            auto f = live.find(id);
            if (f == live.end()) { out += "NOT_FOUND " + to_string(id) + "\n"; continue; }
            Order o = f->second;
            auto lv = book[o.side].find(o.price);
            lv->second.erase(o.it);
            lvl[o.side][o.price] -= o.qty;
            if (lv->second.empty()) { book[o.side].erase(lv); lvl[o.side].erase(o.price); }
            live.erase(f);
            out += "CANCELLED " + to_string(id) + "\n";
        } else if (cmd == "TOP") {
            out += "TOP ";
            if (book[0].empty()) out += "- -";
            else { long long p = prev(book[0].end())->first; out += to_string(p) + " " + to_string(lvl[0][p]); }
            out += " ";
            if (book[1].empty()) out += "- -";
            else { long long p = book[1].begin()->first; out += to_string(p) + " " + to_string(lvl[1][p]); }
            out += "\n";
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
'''

task(id="cpp_write_orderbook", lang="cpp", category="write",
     title="Limit order book matching engine",
     spec=ORDERBOOK_SPEC, tests=orderbook_tests, solution=ORDERBOOK_SOLUTION)


# ---- C#: spreadsheet evaluator --------------------------------------------

SHEET_SPEC = r"""Write a small spreadsheet evaluator.

INPUT
Line 1: N (1 <= N <= 3000). Next N lines: one assignment each, written CELL=EXPRESSION without spaces (for example  B2=A1+7-C3 ). A cell name is one column letter A-Z followed by a row number 1-99 (no leading zeros). EXPRESSION is one or more terms separated by '+' or '-'; a term is a cell name or a non-negative decimal integer of at most 9 digits. A later assignment to the same cell replaces the earlier one.

EVALUATION
- An expression is evaluated from left to right (10-4-3 is 3) with 64-bit signed integers (no test overflows).
- A cell may refer to cells assigned later in the input. A cell that was never assigned has the value 0.
- A cell whose expression depends on itself, directly or through other cells, has the value #CYCLE!. A cell that refers to a #CYCLE! cell is itself #CYCLE!.

OUTPUT
One line per assigned cell (each cell once), sorted by column letter and then by row NUMBER (A2 comes before A10), in the form CELL=VALUE, where VALUE is the integer or #CYCLE!."""

_SHEET_LIMIT = 1 << 60
_SHEET_CYCLE = "#CYCLE!"


class _SheetOverflow(Exception):
    pass


def _sheet_values(inp):
    """Value of every assigned cell: an int or _SHEET_CYCLE."""
    lines = inp.split("\n")
    n = int(lines[0])
    cells = {}
    for ln in lines[1:1 + n]:
        name, _, src = ln.partition("=")
        cells[name] = re.findall(r"[+-]|[A-Z]\d+|\d+", src)
    memo = {}
    busy = set()

    def cell(name):
        if name in memo:
            return memo[name]
        if name not in cells:
            return 0
        if name in busy:
            return _SHEET_CYCLE
        busy.add(name)
        v = ev(cells[name])
        busy.discard(name)
        memo[name] = v
        return v

    def ev(toks):
        total, sign, cyc = 0, 1, False
        for t in toks:
            if t == "+":
                sign = 1
            elif t == "-":
                sign = -1
            else:
                v = int(t) if t[0].isdigit() else cell(t)
                if v == _SHEET_CYCLE:
                    cyc = True
                else:
                    total += sign * v
                    if abs(total) >= _SHEET_LIMIT:
                        raise _SheetOverflow()
        return _SHEET_CYCLE if cyc else total

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(200000)
    try:
        return {name: cell(name) for name in cells}
    finally:
        sys.setrecursionlimit(old)


def ref_sheet(inp):
    vals = _sheet_values(inp)
    names = sorted(vals, key=lambda c: (c[0], int(c[1:])))
    return "".join("%s=%s\n" % (name, vals[name]) for name in names)


def _sheet_random(seed, ncols, nrows, n):
    r = Rng(seed)
    order = [chr(65 + c) + str(row) for c in range(ncols) for row in range(1, nrows + 1)]
    r.shuffle(order)
    rank = {c: i for i, c in enumerate(order)}

    def ref_cell(target):
        # Mostly reference cells earlier in a hidden order (acyclic), sometimes anything.
        if r.chance(88) and rank[target] > 0:
            return order[r.randint(0, rank[target] - 1)]
        return r.choice(order)

    lines = []
    for _ in range(n):
        target = r.choice(order)
        parts = []
        for i in range(r.randint(1, 4)):
            if i:
                parts.append(r.choice("+-"))
            parts.append(str(r.randint(0, 30)) if r.chance(40) else ref_cell(target))
        lines.append("%s=%s" % (target, "".join(parts)))
    return "%d\n%s\n" % (len(lines), "\n".join(lines))


def sheet_tests():
    def mk(*assigns):
        return "%d\n%s\n" % (len(assigns), "\n".join(assigns))

    def cell_name(i):
        return "%s%d" % (chr(65 + i // 99), i % 99 + 1)

    chain = ["A1=1"] + ["%s=%s+1" % (cell_name(i), cell_name(i - 1)) for i in range(1, 400)]
    # Every cell uses the next one three times and the output starts at the top: without
    # memoization the evaluation takes 3^59 steps. All values are 1.
    shared = ["A60=1"] + ["A%d=A%d+A%d-A%d" % (i, i + 1, i + 1, i + 1) for i in range(59, 0, -1)]
    # Each test needs as few features as possible: constants first, then references, then cycles.
    cases = [
        ("constants", mk("A1=7", "B1=0", "C1=123456789", "D1=42")),
        ("addition and subtraction", mk("A1=1+2", "A2=10-4-3", "A3=5-9", "A4=100+20-3+4-50",
                                        "A5=0-0", "A6=999999999+999999999+999999999-1", "B1=8-8-8")),
        ("references to earlier cells", mk("A1=5", "B2=A1+A1", "C3=A1+B2", "D4=C3-B2-A1", "E5=0-A1",
                                           "F6=A1+B2+C3+7")),
        ("forward references", mk("C3=A1+B2", "B2=A1+A1", "A1=5", "D1=C3-B2", "E1=E2", "E2=40+2")),
        ("unassigned cells", mk("A1=Z99+1", "A2=B7-B8+3", "A3=A1+C5", "B1=A3-A2", "C1=0-Q10")),
        ("output order", mk("A10=1", "A2=2", "B11=5", "Z1=4", "A1=3", "B2=6", "A9=9", "M5=8",
                            "A99=7", "B1=0")),
        ("reassigned cells", mk("A1=1", "A2=A1+1", "A1=10", "B1=A2", "B1=A2+A2", "A3=5", "A3=A3",
                                "A3=6")),
        ("cycle: a cell that refers to itself", mk("A1=A1", "B1=5", "C1=C1+1", "D1=B1+B1",
                                                   "E1=E1-E1")),
        ("cycles through other cells", mk("B1=B2", "B2=B3+1", "B3=B1+2", "C1=B2+5", "C2=7",
                                          "C3=C2+C2", "G1=C1-C1", "H1=B1-B1+3", "D1=D2", "D2=D1",
                                          "E1=C3+1")),
        ("shared references", mk(*shared)),
        ("long chain", mk(*chain)),
        ("long chain, reverse order", mk(*list(reversed(chain)))),
    ]
    tests = [T(name, inp, ref_sheet(inp)) for name, inp in cases]
    # The first seeds whose sheets mix numbers and cycles (10-60% #CYCLE!) and stay small.
    seed = 300
    for ncols, nrows, n in [(4, 6, 40), (10, 20, 600), (26, 60, 3000)]:
        while True:
            seed += 1
            inp = _sheet_random(seed, ncols, nrows, n)
            try:
                vals = _sheet_values(inp)
            except _SheetOverflow:
                continue
            share = sum(v == _SHEET_CYCLE for v in vals.values()) / len(vals)
            if 0.1 <= share <= 0.6:
                break
        tests.append(T("random sheet %d assignments" % n, inp, ref_sheet(inp)))
    return tests


SHEET_SOLUTION = r'''using System;
using System.Collections.Generic;
using System.Text;

class Program {
    static Dictionary<string, string> cells = new Dictionary<string, string>();
    static Dictionary<string, long> memo = new Dictionary<string, long>();
    static HashSet<string> busy = new HashSet<string>();
    const long CYC = long.MinValue;

    static long Get(string name) {
        long v;
        if (memo.TryGetValue(name, out v)) return v;
        string src;
        if (!cells.TryGetValue(name, out src)) return 0;
        if (busy.Contains(name)) return CYC;
        busy.Add(name);
        v = Eval(src);
        busy.Remove(name);
        memo[name] = v;
        return v;
    }

    static long Eval(string s) {
        long total = 0, sign = 1;
        bool cyc = false;
        int p = 0;
        while (p < s.Length) {
            char c = s[p];
            if (c == '+' || c == '-') { sign = c == '+' ? 1 : -1; p++; continue; }
            int start = p++;
            while (p < s.Length && char.IsDigit(s[p])) p++;
            string tok = s.Substring(start, p - start);
            long v = char.IsLetter(c) ? Get(tok) : long.Parse(tok);
            if (v == CYC) cyc = true; else total += sign * v;
        }
        return cyc ? CYC : total;
    }

    static void Main() {
        int n = int.Parse(Console.ReadLine().Trim());
        for (int i = 0; i < n; i++) {
            string line = Console.ReadLine().Trim();
            int eq = line.IndexOf('=');
            cells[line.Substring(0, eq)] = line.Substring(eq + 1);
        }
        var names = new List<string>(cells.Keys);
        names.Sort((x, y) => x[0] != y[0] ? x[0].CompareTo(y[0])
                                          : int.Parse(x.Substring(1)).CompareTo(int.Parse(y.Substring(1))));
        var sb = new StringBuilder();
        foreach (string name in names) {
            long v = Get(name);
            sb.Append(name).Append('=').Append(v == CYC ? "#CYCLE!" : v.ToString()).Append('\n');
        }
        Console.Write(sb.ToString());
    }
}
'''

task(id="cs_write_sheet", lang="csharp", category="write",
     title="Spreadsheet evaluator with references and cycles",
     spec=SHEET_SPEC, tests=sheet_tests, solution=SHEET_SOLUTION)


# ---- PHP: RFC 4180 CSV aggregation ----------------------------------------

CSV_SPEC = r"""Write a program that aggregates a CSV file.

INPUT (standard input): CSV text.
- One record per line (lines end with LF). Completely empty lines are ignored.
- Fields are separated by commas. A field may be enclosed in double quotes; a quoted field may contain commas and double quotes, where a literal double quote is written as two double quotes (""). No field contains a line break or a backslash.
- Spaces are significant and are never trimmed.
The first record is the header. It always contains a column named category and a column named amount (at any position, possibly among other columns).
A data record is valid when it has exactly as many fields as the header; any other data record is counted as skipped and otherwise ignored.
In valid records amount matches -?[0-9]+(\.[0-9]{1,2})? with at most 9 integer digits.

OUTPUT
One line per distinct category (categories are compared as exact strings; the empty string is a valid category), in the form  category,count,total : the category, the number of valid records having it, and the sum of their amounts printed with exactly two decimals (examples: 12.50, -0.05, 0.00).
Lines are sorted by total descending, then by category ascending (byte order).
In the output a category is enclosed in double quotes (with inner double quotes doubled) if and only if it contains a comma or a double quote.
After the last category print one final line  skipped=N  where N is the number of skipped records. There is no header line in the output."""


def _csv_parse(text):
    records = []
    field, rec = [], []
    quoted_rec = False
    inq = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if inq:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                inq = False
            else:
                field.append(c)
            i += 1
            continue
        if c == '"':
            inq = True
            quoted_rec = True
        elif c == ",":
            rec.append("".join(field))
            field = []
        elif c == "\n" or c == "\r":
            if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 1
            rec.append("".join(field))
            field = []
            if not (len(rec) == 1 and rec[0] == "" and not quoted_rec):
                records.append(rec)
            rec = []
            quoted_rec = False
        else:
            field.append(c)
        i += 1
    if field or rec or quoted_rec:
        rec.append("".join(field))
        records.append(rec)
    return records


def _cents(s):
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    whole, _, frac = s.partition(".")
    v = int(whole) * 100 + int((frac + "00")[:2])
    return -v if neg else v


def _fmt_cents(v):
    sign = "-" if v < 0 else ""
    v = abs(v)
    return "%s%d.%02d" % (sign, v // 100, v % 100)


def ref_csv(inp):
    records = _csv_parse(inp)
    header = records[0]
    ci, ai = header.index("category"), header.index("amount")
    totals, counts, skipped = {}, {}, 0
    for rec in records[1:]:
        if len(rec) != len(header):
            skipped += 1
            continue
        cat = rec[ci]
        totals[cat] = totals.get(cat, 0) + _cents(rec[ai])
        counts[cat] = counts.get(cat, 0) + 1
    out = []
    for cat in sorted(totals, key=lambda c: (-totals[c], c.encode("utf-8"))):
        name = cat
        if any(ch in cat for ch in ',"\r\n'):
            name = '"' + cat.replace('"', '""') + '"'
        out.append("%s,%d,%s" % (name, counts[cat], _fmt_cents(totals[cat])))
    out.append("skipped=%d" % skipped)
    return "\n".join(out) + "\n"


def _csv_random(seed, n):
    r = Rng(seed)
    cats = ["food", "rent", "Food", "tools, small", 'the "big" one', "", " food",
            "caf\u00e9", "x" * 40, "taxes", "rent "]
    ncols = r.randint(3, 6)
    cols = ["c%d" % i for i in range(ncols)]
    ci, ai = r.randint(0, ncols - 1), r.randint(0, ncols - 1)
    while ai == ci:
        ai = r.randint(0, ncols - 1)
    cols[ci], cols[ai] = "category", "amount"

    def q(s):
        if any(ch in s for ch in ',"') or (s and r.chance(15)):
            return '"' + s.replace('"', '""') + '"'
        return s

    lines = [",".join(cols)]
    for _ in range(n):
        x = r.randint(1, 100)
        if x <= 5:
            lines.append("")
            continue
        fields = []
        for j in range(ncols):
            if j == ci:
                fields.append(q(r.choice(cats)))
            elif j == ai:
                amt = str(r.randint(0, 99999))
                if r.chance(60):
                    amt += "." + ("%d" % r.randint(0, 9) if r.chance(30) else "%02d" % r.randint(0, 99))
                if r.chance(25):
                    amt = "-" + amt
                fields.append(q(amt) if r.chance(10) else amt)
            else:
                fields.append(q(r.choice(["note", "a,b", 'say "hi"', "", "0"])))
        if x <= 13:
            if r.chance(50):
                fields.append("extra")
            else:
                fields.pop()
        lines.append(",".join(fields))
    return "\n".join(lines) + "\n"


def csv_tests():
    big = ["id,amount,category"]
    for i in range(60):
        big.append("%d,%s,%s" % (i, "123456789.%02d" % (i % 100), "huge" if i % 2 else "huge2"))
    big += ["x,0.01,cent"] * 7 + ["y,0.1,dime"] * 3 + ["z,-999999999.99,neg"] * 2
    # Totals are all different unless a test is about ties, and only the later tests have
    # quotes, skipped records or blank lines: one missing feature costs only its own tests.
    cases = [
        ("one record per category", "category,amount\nrent,700.00\nfood,10.50\ntools,0.25\n"),
        ("counts and sums", "category,amount\nfood,10.50\nrent,700.00\nfood,2.25\ntools,0.05\n"
                            "food,1.25\nrent,0.50\n"),
        ("two-decimal formatting", "category,amount\na,700\nb,10.5\nc,3\nc,0.25\nd,0.1\nd,0.2\n"
                                   "e,0.07\n"),
        ("column positions and extra columns",
         "date,amount,note,category,id\n2026-01-01,5,x,books,1\n2026-01-02,7.5,y,games,2\n"
         "2026-01-03,1.25,z,books,3\n"),
        ("negative and zero totals", "category,amount\na,5\nc,-0.05\nd,0.05\nd,-0.05\ne,-3\n"
                                     "f,-0.5\nf,0.25\ng,-10\ng,12.5\n"),
        ("ties are ordered by category", "category,amount\nb,5\na,5\nB,5\nab,5\nc,2.5\nc,2.5\n"
                                         "z,9\nd,1\n"),
        ("skipped records", "category,amount,x\nfood,1,a\nfood,2\nrent,5,b,c\nrent,3,d\nonly\n"
                            "food,4,e\n"),
        ("blank lines", "category,amount\n\nfood,1\n\n\nrent,3\nfood,4\n\n"),
        ("quoted fields with commas",
         'category,amount,note\n"tools, small",3.10,x\nplain,1,"a,b,c"\n"plain",2,"y"\n'
         '"a,b",0.01,z\n'),
        ("doubled quotes",
         'category,amount\n"the ""big"" one",4\n"""quoted""",9.99\n"a,""b"",c",0.01\nplain,1\n'),
        ("spaces and empty category",
         'category,amount\n food,1\nfood,2\nfood ,4\n,8\n"",16\n" ",32\n"food",64\n'),
        ("quoted header and quoted amounts", '"amount","category"\n"7.00","x"\n3,"y"\n"0.5",x\n'),
        ("large sums", "\n".join(big) + "\n"),
    ]
    tests = [T(name, inp, ref_csv(inp)) for name, inp in cases]
    for i, n in enumerate([40, 3000]):
        inp = _csv_random(401 + i, n)
        tests.append(T("random %d records" % n, inp, ref_csv(inp)))
    return tests


CSV_SOLUTION = r'''<?php
$text = stream_get_contents(STDIN);
$records = [];
$field = '';
$rec = [];
$quotedRec = false;
$inq = false;
$n = strlen($text);
for ($i = 0; $i < $n; $i++) {
    $c = $text[$i];
    if ($inq) {
        if ($c === '"') {
            if ($i + 1 < $n && $text[$i + 1] === '"') { $field .= '"'; $i++; }
            else { $inq = false; }
        } else {
            $field .= $c;
        }
        continue;
    }
    if ($c === '"') { $inq = true; $quotedRec = true; }
    elseif ($c === ',') { $rec[] = $field; $field = ''; }
    elseif ($c === "\n" || $c === "\r") {
        if ($c === "\r" && $i + 1 < $n && $text[$i + 1] === "\n") { $i++; }
        $rec[] = $field; $field = '';
        if (!(count($rec) === 1 && $rec[0] === '' && !$quotedRec)) { $records[] = $rec; }
        $rec = []; $quotedRec = false;
    } else { $field .= $c; }
}
if ($field !== '' || count($rec) > 0 || $quotedRec) { $rec[] = $field; $records[] = $rec; }

$header = $records[0];
$ci = array_search('category', $header, true);
$ai = array_search('amount', $header, true);
$totals = []; $counts = []; $names = []; $skipped = 0;
for ($k = 1; $k < count($records); $k++) {
    $r = $records[$k];
    if (count($r) !== count($header)) { $skipped++; continue; }
    $key = 'k' . $r[$ci];
    $s = $r[$ai];
    $neg = false;
    if ($s !== '' && $s[0] === '-') { $neg = true; $s = substr($s, 1); }
    $parts = explode('.', $s);
    $frac = isset($parts[1]) ? substr($parts[1] . '00', 0, 2) : '00';
    $cents = intval($parts[0]) * 100 + intval($frac);
    if ($neg) { $cents = -$cents; }
    if (!isset($totals[$key])) { $totals[$key] = 0; $counts[$key] = 0; $names[$key] = $r[$ci]; }
    $totals[$key] += $cents; $counts[$key]++;
}
$keys = array_keys($totals);
usort($keys, function ($a, $b) use ($totals, $names) {
    if ($totals[$a] !== $totals[$b]) { return $totals[$a] > $totals[$b] ? -1 : 1; }
    return strcmp($names[$a], $names[$b]);
});
$out = "";
foreach ($keys as $key) {
    $cat = $names[$key];
    if (strpbrk($cat, ",\"\r\n") !== false) { $cat = '"' . str_replace('"', '""', $cat) . '"'; }
    $v = $totals[$key];
    $sign = $v < 0 ? '-' : '';
    $v = abs($v);
    $out .= $cat . ',' . $counts[$key] . ',' . $sign . intdiv($v, 100) . '.' . sprintf('%02d', $v % 100) . "\n";
}
echo $out . "skipped=" . $skipped . "\n";
'''

task(id="php_write_csv", lang="php", category="write",
     title="CSV parser and aggregation",
     spec=CSV_SPEC, tests=csv_tests, solution=CSV_SOLUTION)


# The reference and the solution also implement a power operator (^) that the task no longer has.
# ---- Java: exact rational calculator --------------------------------------

RATIONAL_SPEC = r"""Write an exact rational-number calculator.

INPUT
Lines until end of input (at most 2000 lines, each at most 2000 characters). Each line holds one expression. Spaces and tabs may appear anywhere between tokens. Grammar:
  expr  := term (('+' | '-') term)*
  term  := unary (('*' | '/') unary)*
  unary := '-' unary | atom
  atom  := INTEGER | '(' expr ')'
INTEGER is a sequence of 1 to 40 decimal digits. Every line is syntactically valid.

SEMANTICS
- Exact arithmetic on rational numbers of unlimited size; '/' is exact division (1/3 stays one third).
- A division by zero anywhere in the expression makes the result of the whole line ERROR. The following lines are still evaluated.

OUTPUT
One line per expression: ERROR, or the result as a fraction in lowest terms p/q with q > 0, printed as just p when q = 1 (examples: -7/3, 5, 0)."""


class _RatError(Exception):
    pass


def _rat_eval(src):
    toks = re.findall(r"\d+|[-+*/^()]", src)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def expr():
        v = term()
        while peek() in ("+", "-"):
            if take() == "+":
                v = v + term()
            else:
                v = v - term()
        return v

    def term():
        v = unary()
        while peek() in ("*", "/"):
            if take() == "*":
                v = v * unary()
            else:
                d = unary()
                if d == 0:
                    raise _RatError()
                v = v / d
        return v

    def unary():
        if peek() == "-":
            take()
            return -unary()
        return power()

    def power():
        base = atom()
        if peek() == "^":
            take()
            e = unary()
            assert e.denominator == 1 and abs(e) <= 64, "bad generated exponent"
            e = int(e)
            if e < 0 and base == 0:
                raise _RatError()
            return base ** e
        return base

    def atom():
        t = take()
        if t == "(":
            v = expr()
            take()
            return v
        return Fraction(int(t))

    return expr()


def ref_rational(inp):
    out = []
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(200000)
    try:
        for line in inp.split("\n"):
            if not line.strip():
                continue
            try:
                v = _rat_eval(line)
                out.append(str(v.numerator) if v.denominator == 1
                           else "%d/%d" % (v.numerator, v.denominator))
            except _RatError:
                out.append("ERROR")
    finally:
        sys.setrecursionlimit(old)
    return "\n".join(out) + "\n"


def _rat_random(seed, n, big=False):
    r = Rng(seed)

    def num():
        if big and r.chance(40):
            return str(r.randint(1, 9)) + "".join(str(r.randint(0, 9)) for _ in range(r.randint(10, 39)))
        return str(r.randint(0, 12))

    def sp():
        return r.choice(["", "", "", " ", "  ", "\t"])

    def gen(depth):
        x = r.randint(1, 100)
        if depth == 0 or x <= 25:
            return num()
        if x <= 37:
            return "-" + sp() + gen(depth - 1)
        if x <= 52:
            return "(" + sp() + gen(depth - 1) + sp() + ")"
        op = r.choice(["+", "-", "*", "/"])
        return gen(depth - 1) + sp() + op + sp() + gen(depth - 1)

    return "\n".join(gen(r.randint(2, 5)) for _ in range(n)) + "\n"


def rational_tests():
    deep = "(" * 60 + "1" + ")" * 60
    deep2 = "1" + "".join("+(2" for _ in range(60)) + ")" * 60
    # From integers only to fractions, signs and errors: each test adds one feature.
    cases = [
        ("integer results", "1+2\n7-3\n3*4\n8/4\n0\n12345\n100/5/2\n6*7-2\n"),
        ("precedence and parentheses", "1+2*3\n(1+2)*3\n10-4-3\n10-(4-3)\n2*3+4*5\n2*(3+4)*5\n"
                                       "((2))\n100/(5*2)\n100/5*2\n"),
        ("fractions in lowest terms", "1/3\n2/6\n1/3+1/6\n3/4-1/4\n6/4\n1/2*2/3\n1/2/3\n1/(2/3)\n"
                                      "7/7\n0/5\n5/10+5/10\n22/7-3\n"),
        ("negative results", "7-10\n1/3-1/2\n0-5\n2*(3-9)\n(1-4)/6\n3/(2-5)\n(2-3)/(4-7)\n0/(1-2)\n"),
        ("unary minus", "-1/2\n1/-2\n-1/-2\n-(1/2)\n--3\n3--3\n3-(-3)\n-0\n-(3-5)/4\n2*-3\n-2*-3\n"
                        "---5\n"),
        ("division by zero", "1/0\n1/(2-2)\n5+1/0*0\n0*(1/0)\n1/(1/3-2/6)\n(3-3)/(3-3)\n-(1/0)\n"),
        ("lines after an error", "1/0\n7\n1/2\n2/(1-1)\n3*3\n0/0\n0/1\n"),
        ("big integers", "1234567890123456789012345678901234567890+1\n"
                         "9999999999999999999999999999999999999999*9999999999999999999999999999999999999999\n"
                         "1234567890123456789012345678901234567890/9876543210987654321098765432109876543210\n"
                         "18446744073709551616*18446744073709551616\n"
                         "9223372036854775807+1\n4294967296*4294967296\n"
                         "1/3+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53+1/59\n"),
        ("whitespace", "  1 +\t2  \n( ( 3 ) )\n 4 /\t( 2 * ( 1 + 1 ) ) \n\t-\t(\t5\t)\t*\t2\n"
                       "1   /   3\n"),
        ("deep nesting", deep + "\n" + deep2 + "\n" + "-" * 40 + "7\n"),
    ]
    tests = [T(name, inp, ref_rational(inp)) for name, inp in cases]
    for i, (n, big) in enumerate([(60, False), (300, True), (2000, True)]):
        inp = _rat_random(501 + i, n, big)
        tests.append(T("random %d expressions%s" % (n, " (large integers)" if big else ""),
                       inp, ref_rational(inp)))
    return tests


RATIONAL_SOLUTION = r'''import java.io.*;
import java.math.BigInteger;

public class Main {
    static String s; static int p;
    static class Zero extends RuntimeException {}
    static class Q {
        BigInteger n, d;
        Q(BigInteger n, BigInteger d) {
            if (d.signum() == 0) throw new Zero();
            if (d.signum() < 0) { n = n.negate(); d = d.negate(); }
            BigInteger g = n.gcd(d);
            if (g.signum() != 0 && !g.equals(BigInteger.ONE)) { n = n.divide(g); d = d.divide(g); }
            this.n = n; this.d = d;
        }
        Q add(Q o) { return new Q(n.multiply(o.d).add(o.n.multiply(d)), d.multiply(o.d)); }
        Q sub(Q o) { return new Q(n.multiply(o.d).subtract(o.n.multiply(d)), d.multiply(o.d)); }
        Q mul(Q o) { return new Q(n.multiply(o.n), d.multiply(o.d)); }
        Q div(Q o) { return new Q(n.multiply(o.d), d.multiply(o.n)); }
        Q neg() { return new Q(n.negate(), d); }
        Q pow(int e) {
            if (e >= 0) return new Q(n.pow(e), d.pow(e));
            return new Q(d.pow(-e), n.pow(-e));
        }
    }
    static void ws() { while (p < s.length() && (s.charAt(p) == ' ' || s.charAt(p) == '\t' || s.charAt(p) == '\r')) p++; }
    static char peek() { ws(); return p < s.length() ? s.charAt(p) : '\0'; }
    static Q expr() {
        Q v = term();
        while (true) {
            char c = peek();
            if (c == '+') { p++; v = v.add(term()); }
            else if (c == '-') { p++; v = v.sub(term()); }
            else return v;
        }
    }
    static Q term() {
        Q v = unary();
        while (true) {
            char c = peek();
            if (c == '*') { p++; v = v.mul(unary()); }
            else if (c == '/') { p++; v = v.div(unary()); }
            else return v;
        }
    }
    static Q unary() {
        if (peek() == '-') { p++; return unary().neg(); }
        return power();
    }
    static Q power() {
        Q b = atom();
        if (peek() == '^') { p++; Q e = unary(); return b.pow(e.n.intValueExact()); }
        return b;
    }
    static Q atom() {
        char c = peek();
        if (c == '(') { p++; Q v = expr(); peek(); p++; return v; }
        int st = p;
        while (p < s.length() && Character.isDigit(s.charAt(p))) p++;
        return new Q(new BigInteger(s.substring(st, p)), BigInteger.ONE);
    }
    public static void main(String[] args) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            s = line; p = 0;
            try {
                Q v = expr();
                out.append(v.d.equals(BigInteger.ONE) ? v.n.toString() : v.n + "/" + v.d).append('\n');
            } catch (Zero z) {
                out.append("ERROR\n");
            }
        }
        System.out.print(out);
    }
}
'''

task(id="java_write_rational", lang="java", category="write",
     title="Exact rational calculator (big integers, errors)",
     spec=RATIONAL_SPEC, tests=rational_tests, solution=RATIONAL_SOLUTION)


# ==========================================================================
# HIDDEN-BUG TASKS (one per language)
#
# Each task ships a program with several independent, language-typical bugs.
# "fixes" (old, new) pairs are only used by --selftest to build a known-good
# version and prove that every hidden test is satisfiable.
# ==========================================================================

def avg_is_tie(total, count):
    """True when total/count has a 5 exactly at the third decimal (rounding to 2 decimals is then ambiguous)."""
    return (total * 1000) % count == 0 and (total * 1000 // count) % 10 == 5


def apply_fixes(src, fixes):
    for old, new in fixes:
        if src.count(old) != 1:
            raise AssertionError("fix anchor not unique: %r" % old)
        src = src.replace(old, new)
    return src


# ---- Python: sales report --------------------------------------------------

SALES_SPEC = r"""Input: line 1 holds N (1 <= N <= 100000). Each of the next N lines is one sale: store_id item_code units unit_price (separated by spaces). store_id is a positive integer without leading zeros; units and unit_price are positive integers up to 10^12. The last line holds M followed by M integer thresholds.

Output:
1. One line per store, in ascending NUMERIC order of store_id:
   store <id>: sales=<number of sales> revenue=<sum of units*unit_price> avg=<average revenue per sale rounded to the nearest integer, exact halves rounded up>
2. One line per threshold, in input order:
   >= <t>: <number of stores whose revenue is at least t>
3. One line:
   best: <ids of all stores having the maximum revenue, in ascending numeric order, separated by single spaces>
All results must be exact for the whole input range."""

SALES_GIVEN = r'''import sys


def collect(record, batch=[]):
    batch.append(record)
    return batch


def summarize(batch):
    count = len(batch)
    revenue = 0
    for units, price in batch:
        revenue += units * price
    average = round(revenue / count)
    return count, revenue, average


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0])
    by_store = {}
    for line in data[1:1 + n]:
        store, item, units, price = line.split()
        if store in by_store:
            by_store[store] = collect((int(units), int(price)), by_store[store])
        else:
            by_store[store] = collect((int(units), int(price)))
    parts = data[1 + n].split()
    thresholds = [int(x) for x in parts[1:1 + int(parts[0])]]

    revenue_of = {}
    for store in sorted(by_store):
        count, revenue, average = summarize(by_store[store])
        revenue_of[store] = revenue
        print("store %s: sales=%d revenue=%d avg=%d" % (store, count, revenue, average))

    checks = []
    for t in thresholds:
        checks.append(lambda rev: rev >= t)
    for limit, check in zip(thresholds, checks):
        print(">= %d: %d" % (limit, sum(1 for rev in revenue_of.values() if check(rev))))

    best = max(revenue_of.values())
    winners = [s for s in sorted(revenue_of) if revenue_of[s] is best]
    print("best: " + " ".join(winners))


main()
'''

SALES_FIXES = [
    ("def collect(record, batch=[]):\n    batch.append(record)",
     "def collect(record, batch=None):\n    if batch is None:\n        batch = []\n    batch.append(record)"),
    ("average = round(revenue / count)", "average = (2 * revenue + count) // (2 * count)"),
    ("for store in sorted(by_store):", "for store in sorted(by_store, key=int):"),
    ("checks.append(lambda rev: rev >= t)", "checks.append(lambda rev, t=t: rev >= t)"),
    ("[s for s in sorted(revenue_of) if revenue_of[s] is best]",
     "[s for s in sorted(revenue_of, key=int) if revenue_of[s] == best]"),
]


def ref_sales(inp):
    data = inp.split("\n")
    n = int(data[0])
    stores = {}
    for line in data[1:1 + n]:
        s, _, u, p = line.split()
        stores.setdefault(int(s), []).append(int(u) * int(p))
    parts = data[1 + n].split()
    thresholds = [int(x) for x in parts[1:1 + int(parts[0])]]
    out = []
    rev = {}
    for s in sorted(stores):
        total, cnt = sum(stores[s]), len(stores[s])
        rev[s] = total
        out.append("store %d: sales=%d revenue=%d avg=%d" % (s, cnt, total,
                                                             (2 * total + cnt) // (2 * cnt)))
    for t in thresholds:
        out.append(">= %d: %d" % (t, sum(1 for v in rev.values() if v >= t)))
    best = max(rev.values())
    out.append("best: " + " ".join(str(s) for s in sorted(rev) if rev[s] == best))
    return "\n".join(out) + "\n"


def _sales_input(sales, thresholds):
    return "%d\n%s\n%d %s\n" % (len(sales), "\n".join("%s %s %d %d" % s for s in sales),
                                len(thresholds), " ".join(str(t) for t in thresholds))


def _sales_random(seed, n, nstores):
    r = Rng(seed)
    ids = [str(r.randint(1, 3000)) for _ in range(nstores)]
    sales = []
    for _ in range(n):
        big = r.chance(10)
        sales.append((r.choice(ids), "i%d" % r.randint(1, 50),
                      r.randint(1, 10 ** 12 if big else 40), r.randint(1, 10 ** 12 if big else 500)))
    # Force a tie for the maximum between two fresh stores.
    top = sum(u * p for _, _, u, p in sales) + 12345
    sales.append(("3500", "tie", 1, top))
    sales.append(("900", "tie", 1, top - 7))
    sales.append(("900", "tie", 7, 1))
    r.shuffle(sales)
    ths = [r.randint(1, 10 ** 6) for _ in range(5)] + [top, top + 1]
    return _sales_input(sales, ths)


def sales_tests():
    h = 10 ** 12
    cases = [
        ("single store", _sales_input([("3", "pen", 2, 10), ("3", "ink", 1, 40), ("3", "cap", 3, 20)], [50])),
        ("single store, average rounds down", _sales_input([("7", "a", 1, 3), ("7", "b", 1, 3), ("7", "c", 1, 4)], [10])),
        ("several stores", _sales_input([("1", "a", 2, 10), ("2", "b", 1, 7), ("1", "c", 1, 13),
                                         ("3", "d", 4, 25), ("2", "e", 2, 2)], [30])),
        ("store ids of different lengths", _sales_input([("10", "a", 1, 5), ("9", "b", 1, 6), ("100", "c", 1, 7),
                                                         ("2", "d", 1, 8), ("21", "e", 1, 9)], [6])),
        ("average exactly half (2.5)", _sales_input([("5", "a", 1, 2), ("5", "b", 1, 3)], [1])),
        ("average exactly half (4.5, 0.5-like)", _sales_input([("5", "a", 3, 3), ("5", "b", 1, 1), ("5", "c", 2, 4),
                                                              ("5", "d", 1, 0 + 1), ("5", "e", 1, 2), ("5", "f", 6, 1)], [1])),
        ("huge exact average", _sales_input([("8", "a", h, h), ("8", "b", 1, 1)], [1])),
        ("huge average, thirds", _sales_input([("8", "a", h, h - 1), ("8", "b", h - 7, h), ("8", "c", 3, 5)], [1])),
        ("several thresholds", _sales_input([("4", "a", 10, 10)], [99, 100, 101, 0])),
        ("tie for best", _sales_input([("1", "a", 10, 100), ("2", "b", 500, 2), ("3", "c", 1, 999),
                                       ("2", "d", 0 + 1, 0 + 1), ("1", "e", 1, 1)], [1000])),
        ("tie for best, three stores", _sales_input([("30", "a", 1, 70000), ("4", "b", 7, 10000), ("200", "c", 2, 35000),
                                                     ("5", "d", 1, 1)], [70000, 70001])),
    ]
    tests = [T(name, inp, ref_sales(inp)) for name, inp in cases]
    for i, (n, st) in enumerate([(40, 6), (400, 25), (20000, 300)]):
        inp = _sales_random(601 + i, n, st)
        tests.append(T("random %d sales" % n, inp, ref_sales(inp)))
    return tests


task(id="py_bugfix_sales", lang="python", category="bugfix",
     title="Sales report with hidden bugs",
     spec=SALES_SPEC, given=SALES_GIVEN, fixes=SALES_FIXES, tests=sales_tests)


# ---- C: word frequency -----------------------------------------------------

WORDS_SPEC = r"""Input: the first token is an integer K (0 <= K <= 1000). Everything after it, up to end of input, is arbitrary text (any byte values).
A word is a maximal run of ASCII letters (A-Z, a-z); every other byte is a separator. Words are case-insensitive and are reported in lower case. A word longer than 31 letters is truncated to its first 31 letters (the extra letters are dropped; they do not start a new word), and all statistics use the truncated word.

Output:
Line 1:  words=<total number of words> unique=<number of distinct words> avglen=<average word length with exactly 2 decimals; 0.00 when there is no word>
Then the K most frequent words (fewer if there are not that many), one per line as  <word> <count> , ordered by count descending and, for equal counts, by word ascending (strcmp order).
The number of distinct words is not limited. The program must not have memory errors (it is also run under AddressSanitizer)."""

WORDS_GIVEN = r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAXWORD 32

typedef struct {
    char word[MAXWORD];
    int count;
} Entry;

static Entry *table = NULL;
static int used = 0, cap = 0;

static void add_word(const char *w) {
    for (int i = 0; i < used; i++) {
        if (strcmp(table[i].word, w) == 0) {
            table[i].count++;
            return;
        }
    }
    if (used == cap) {
        cap *= 2;
        table = realloc(table, cap * sizeof(table));
        if (!table) exit(1);
    }
    strcpy(table[used].word, w);
    table[used].count = 1;
    used++;
}

static int by_count(const void *a, const void *b) {
    const Entry *x = a, *y = b;
    return y->count - x->count;
}

int main(void) {
    int k;
    if (scanf("%d", &k) != 1) return 1;

    cap = 16;
    table = malloc(cap * sizeof(Entry));
    if (!table) return 1;

    char word[MAXWORD];
    int len = 0;
    long total_words = 0, total_len = 0;
    char c;
    while ((c = getchar()) != EOF) {
        if (isalpha((unsigned char)c)) {
            if (len < MAXWORD) word[len++] = (char)tolower((unsigned char)c);
        } else if (len > 0) {
            word[len] = '\0';
            add_word(word);
            total_words++;
            total_len += len;
            len = 0;
        }
    }

    double avg = total_words > 0 ? total_len / total_words : 0;
    printf("words=%ld unique=%d avglen=%.2f\n", total_words, used, avg);

    qsort(table, used, sizeof(Entry), by_count);
    for (int i = 0; i < k && i < used; i++)
        printf("%s %d\n", table[i].word, table[i].count);

    free(table);
    return 0;
}
'''

WORDS_FIXES = [
    ("cap * sizeof(table)", "cap * sizeof(Entry)"),
    ("    return y->count - x->count;",
     "    if (y->count != x->count) return y->count > x->count ? 1 : -1;\n    return strcmp(x->word, y->word);"),
    ("    char c;\n", "    int c;\n"),
    ("if (len < MAXWORD) word", "if (len < MAXWORD - 1) word"),
    ("    double avg = total_words > 0 ? total_len / total_words : 0;",
     "    if (len > 0) {\n        word[len] = '\\0';\n        add_word(word);\n        total_words++;\n"
     "        total_len += len;\n    }\n"
     "    double avg = total_words > 0 ? (double)total_len / (double)total_words : 0;"),
]


def ref_words(data):
    m = re.match(rb"\s*(\d+)", data)
    k = int(m.group(1))
    words = [w.lower()[:31] for w in re.findall(rb"[A-Za-z]+", data[m.end():])]
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    total = len(words)
    avg = (sum(len(w) for w in words) / total) if total else 0.0
    ref_words.tie = bool(total) and avg_is_tie(sum(len(w) for w in words), total)
    out = ["words=%d unique=%d avglen=%.2f" % (total, len(counts), avg)]
    for w in sorted(counts, key=lambda x: (-counts[x], x))[:k]:
        out.append("%s %d" % (w.decode("ascii"), counts[w]))
    return "\n".join(out) + "\n"


def _words_random(seed, n, end_sep):
    r = Rng(seed)
    pool = ["".join(chr(97 + r.randint(0, 25)) for _ in range(r.randint(1, 9))) for _ in range(60)]
    pool += ["supercalifragilisticexpialidociousness", "abcdefghijklmnopqrstuvwxyzabcdeXYZ",
             "abcdefghijklmnopqrstuvwxyzabcdeQ", "a" * 31, "b" * 32, "c" * 64]
    seps = [b" ", b" ", b" ", b"\n", b", ", b". ", b"-", b"'", b"3", b"\t", b" \xff ", b"\xe9", b"\xff"]
    parts = [b"%d\n" % r.randint(5, 40)]
    for _ in range(n):
        w = r.choice(pool)
        if r.chance(30):
            w = w.upper() if r.chance(50) else w.capitalize()
        parts.append(w.encode("ascii"))
        parts.append(r.choice(seps))
    last = r.choice(pool).encode("ascii")
    parts.append(last)
    if end_sep:
        parts.append(b"\n")
    return b"".join(parts)


def words_tests():
    many = " ".join(" ".join(["%c%c%c%c" % (97 + i // 26, 97 + i % 26, 120, 121)] * (i + 1)) for i in range(40))
    long40 = "internationalizationalitiesandmoreletters"   # 41 letters
    w32 = "abcdefghijklmnopqrstuvwxyzabcdef"                # 32 letters
    cases = [
        ("basic", b"3\nTree tree TREE bird Bird lamp\n"),
        ("K is zero", b"0\nrock rock sand\n"),
        ("K larger than vocabulary", b"10\n lake lake lake pond pond; mist!\n"),
        ("ties broken alphabetically", b"6\nzeta yoke xray wolf vase unit\n"),
        ("ties among frequent words", b"4\nzulu zulu yank yank xmas xmas wasp wasp alfa alfa alfa\n"),
        ("more than 16 distinct words", ("5\n" + many + "\n").encode()),
        ("average with decimals", b"3\na a a bb bb cccc\n"),
        ("average with decimals 2", b"2\nx x yy zzzzzzz zzzzzzz zzzzzzz\n"),
        ("word of 32 letters", ("2\n" + w32 + " a a\n").encode()),
        ("very long words share a prefix", ("3\n" + long40 + " " + long40[:31] + "zzzzzzzzzzzzzzzzzzzz b b b " + "q" * 200 + "\n").encode()),
        ("no separator at end of input", b"3\ntree tree tree bird bird lamp lamp lamp lamp"),
        ("single word, no newline", b"1\nhello"),
        ("byte 0xFF inside the text", b"3\nrock rock \xff sand sand sand\xffclay clay clay clay\n"),
        ("empty text", b"5\n"),
    ]
    tests = [T(name, inp, ref_words(inp)) for name, inp in cases]
    seed = 700
    for n, sep in [(120, True), (600, False), (3000, False)]:
        while True:
            seed += 1
            inp = _words_random(seed, n, sep)
            exp = ref_words(inp)
            if not ref_words.tie:
                break
        tests.append(T("random %d words" % n, inp, exp))
    return tests


task(id="c_bugfix_words", lang="c", category="bugfix",
     title="Word frequency counter with hidden bugs",
     spec=WORDS_SPEC, given=WORDS_GIVEN, fixes=WORDS_FIXES, tests=words_tests)


# ---- C#: warehouse stock ---------------------------------------------------

STOCK_SPEC = r"""Input: line 1 holds N (1 <= N <= 100000). Each of the next N lines is one operation:
  IN name qty price     add qty units (0 <= qty <= 10^9) of the item; price (1 <= price <= 10^9) becomes the item's current unit price. A name seen for the first time is appended to the item list.
  OUT name qty          remove qty units (1 <= qty <= 10^9). If the item does not exist or holds fewer than qty units, the operation is rejected and changes nothing.
Names contain no spaces.

Output, after all operations were processed:
1. For every item whose quantity is greater than 0, in order of first appearance:  <name> <qty> <value>  where value = qty * current unit price.
2. One line  avg=<average quantity of the listed items with exactly 2 decimals; 0.00 when no item is listed>
3. For every rejected operation, in input order:  REJECT #<k> <name>  where k is the 1-based position of that operation in the input.
All numbers must be exact; quantities and values can exceed 32 bits (they always fit in 64 bits)."""

STOCK_GIVEN = r'''using System;
using System.Collections.Generic;
using System.Globalization;

struct Item
{
    public string Name;
    public int Qty;
    public int Price;
}

class Program
{
    static void Main()
    {
        int n = int.Parse(Console.ReadLine().Trim());
        string[][] ops = new string[n][];
        for (int i = 0; i < n; i++)
            ops[i] = Console.ReadLine().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        var items = new List<Item>();
        var rejects = new List<Func<string>>();

        for (int i = 0; i < n; i++)
        {
            string[] op = ops[i];
            string name = op[1];
            int idx = items.FindIndex(x => x.Name == name);
            if (op[0] == "IN")
            {
                int qty = int.Parse(op[2]);
                int price = int.Parse(op[3]);
                if (idx < 0)
                {
                    items.Add(new Item { Name = name, Qty = qty, Price = price });
                }
                else
                {
                    Item it = items[idx];
                    it.Qty += qty;
                    it.Price = price;
                }
            }
            else
            {
                int qty = int.Parse(op[2]);
                if (idx < 0 || items[idx].Qty < qty)
                {
                    rejects.Add(() => "REJECT #" + (i + 1) + " " + name);
                }
                else
                {
                    Item it = items[idx];
                    it.Qty -= qty;
                }
            }
        }

        foreach (Item it in items)
        {
            if (it.Qty == 0) items.Remove(it);
        }

        int totalQty = 0;
        foreach (Item it in items)
        {
            int value = it.Qty * it.Price;
            Console.WriteLine(it.Name + " " + it.Qty + " " + value);
            totalQty += it.Qty;
        }
        double avg = items.Count > 0 ? totalQty / items.Count : 0;
        Console.WriteLine("avg=" + avg.ToString("F2", CultureInfo.InvariantCulture));

        foreach (var r in rejects) Console.WriteLine(r());
    }
}
'''

STOCK_FIXES = [
    ("    public int Qty;", "    public long Qty;"),
    ("                    it.Price = price;", "                    it.Price = price;\n                    items[idx] = it;"),
    ("                    it.Qty -= qty;", "                    it.Qty -= qty;\n                    items[idx] = it;"),
    ('                    rejects.Add(() => "REJECT #" + (i + 1) + " " + name);',
     '                    int k = i + 1;\n                    rejects.Add(() => "REJECT #" + k + " " + name);'),
    ("        foreach (Item it in items)\n        {\n            if (it.Qty == 0) items.Remove(it);\n        }",
     "        items.RemoveAll(x => x.Qty == 0);"),
    ("        int totalQty = 0;", "        long totalQty = 0;"),
    ("            int value = it.Qty * it.Price;", "            long value = it.Qty * (long)it.Price;"),
    ("totalQty / items.Count : 0;", "(double)totalQty / items.Count : 0;"),
]


def ref_stock(inp):
    lines = inp.split("\n")
    n = int(lines[0])
    order, qty, price, rejects = [], {}, {}, []
    for k, ln in enumerate(lines[1:1 + n], 1):
        f = ln.split()
        name = f[1]
        if f[0] == "IN":
            if name not in qty:
                order.append(name)
                qty[name] = 0
            qty[name] += int(f[2])
            price[name] = int(f[3])
        else:
            q = int(f[2])
            if name not in qty or qty[name] < q:
                rejects.append("REJECT #%d %s" % (k, name))
            else:
                qty[name] -= q
    listed = [nm for nm in order if qty[nm] > 0]
    out = ["%s %d %d" % (nm, qty[nm], qty[nm] * price[nm]) for nm in listed]
    avg = (sum(qty[nm] for nm in listed) / len(listed)) if listed else 0.0
    ref_stock.tie = bool(listed) and avg_is_tie(sum(qty[nm] for nm in listed), len(listed))
    out.append("avg=%.2f" % avg)
    out.extend(rejects)
    return "\n".join(out) + "\n"


def _stock_input(ops):
    return "%d\n%s\n" % (len(ops), "\n".join(ops))


def _stock_random(seed, n, big):
    r = Rng(seed)
    names = ["bolt", "nut", "gear", "axle", "cog", "pin", "rod", "cam", "hub", "key", "rim", "tab"]
    ops = []
    for _ in range(n):
        nm = r.choice(names)
        if r.chance(55):
            q = r.choice([0, 0, r.randint(1, 50), r.randint(1, 50), 10 ** 9 if big else 7])
            ops.append("IN %s %d %d" % (nm, q, r.randint(1, 10 ** 7 if big else 90)))
        else:
            ops.append("OUT %s %d" % (nm, r.choice([r.randint(1, 60), r.randint(1, 20), 10 ** 9 if big else 3])))
    return _stock_input(ops)


def stock_tests():
    B = 10 ** 9
    cases = [
        ("distinct items only", _stock_input(["IN bolt 2 5", "IN nut 4 3", "IN gear 6 10"])),
        ("single item", _stock_input(["IN cog 7 11"])),
        ("repeated IN updates quantity and price", _stock_input(["IN bolt 5 10", "IN bolt 3 20"])),
        ("OUT reduces quantity", _stock_input(["IN bolt 5 10", "OUT bolt 2"])),
        ("IN and OUT interleaved", _stock_input(["IN a 5 2", "IN b 9 3", "OUT a 1", "IN a 4 7", "OUT b 9", "IN b 3 3",
                                                 "OUT a 8"])),
        ("average with decimals", _stock_input(["IN a 1 1", "IN b 2 1"])),
        ("average with decimals 2", _stock_input(["IN a 1 1", "IN b 1 1", "IN c 5 1"])),
        ("value above 32 bits", _stock_input(["IN a %d %d" % (B, B)])),
        ("quantity above 32 bits", _stock_input(["IN a %d 3" % B] * 5 + ["IN b %d 1" % B] * 3)),
        ("rejected operations keep their position", _stock_input(["IN a 1 1", "OUT b 1", "IN c 3 1"])),
        ("several rejections", _stock_input(["OUT x 1", "IN a 2 2", "IN b 2 2", "IN c 2 2", "OUT a 3", "OUT y 9",
                                             "IN d 2 2"])),
        ("items with zero quantity are not listed", _stock_input(["IN a 0 5", "IN b 2 2"])),
        ("zero quantity in the last position", _stock_input(["IN a 3 3", "IN b 0 2"])),
        ("nothing listed", _stock_input(["IN a 0 5", "OUT a 1", "OUT q 2"])),
    ]
    tests = [T(name, inp, ref_stock(inp)) for name, inp in cases]
    seed = 800
    for n, big in [(40, False), (300, False), (2000, True)]:
        while True:
            seed += 1
            inp = _stock_random(seed, n, big)
            exp = ref_stock(inp)
            if not ref_stock.tie:           # avoid x.xx5 averages (formatting of exact ties is not specified)
                break
        tests.append(T("random %d operations" % n, inp, exp))
    return tests


task(id="cs_bugfix_stock", lang="csharp", category="bugfix",
     title="Warehouse stock report with hidden bugs",
     spec=STOCK_SPEC, given=STOCK_GIVEN, fixes=STOCK_FIXES, tests=stock_tests)


# ---- PHP: order summary ----------------------------------------------------

ORDERS_SPEC = r"""Input: text lines of the form  order_id;customer_id;amount;tag  (fields separated by ';', no field contains ';').
- Lines that do not have exactly 4 fields are ignored. Lines with an empty order_id are ignored.
- order_id is any non-empty string and is compared as an exact string (byte for byte).
- customer_id is a non-negative integer without leading zeros (0, 7, 42, ...).
- amount is a non-negative decimal number with at most two decimals (12, 12.5, 12.50, 0.07), below 10^9.
- tag is free text (possibly empty). An order whose tag contains VOID anywhere, in any letter case, is excluded.
- Among the orders that are not excluded, only the first one with a given order_id is counted; later ones with the same order_id are ignored.

Output:
Line 1:  orders=<number of counted orders> total=<sum of their amounts with exactly two decimals>
Then the (up to) 3 customers with the highest totals, highest first, ties broken by the smaller customer_id, one per line:  customer <customer_id> <total with exactly two decimals>
All sums must be exact to the cent."""

ORDERS_GIVEN = r'''<?php
$lines = explode("\n", stream_get_contents(STDIN));
$orders = [];
foreach ($lines as $line) {
    $line = rtrim($line, "\r");
    if ($line === '') {
        continue;
    }
    $f = explode(';', $line);
    if (count($f) != 4) {
        continue;
    }
    $orders[] = ['id' => $f[0], 'customer' => $f[1], 'amount' => $f[2], 'tag' => $f[3]];
}

// Normalise the tags once.
foreach ($orders as &$o) {
    $o['tag'] = strtoupper(trim($o['tag']));
}

$seen = [];
$totals = [];
$count = 0;
$sum = 0;
foreach ($orders as $o) {
    if (empty($o['id'])) {
        continue;
    }
    if (strpos($o['tag'], 'VOID')) {
        continue;
    }
    if (in_array($o['id'], $seen)) {
        continue;
    }
    $seen[] = $o['id'];
    $cents = (int)($o['amount'] * 100);
    $cid = $o['customer'];
    if (!isset($totals[$cid])) {
        $totals[$cid] = 0;
    }
    $totals[$cid] += $cents;
    $count++;
    $sum += $cents;
}

function money($cents)
{
    return sprintf('%d.%02d', intdiv($cents, 100), $cents % 100);
}

echo "orders=$count total=" . money($sum) . "\n";

uksort($totals, function ($a, $b) use ($totals) {
    if ($totals[$a] != $totals[$b]) {
        return $totals[$b] <=> $totals[$a];
    }
    return $a <=> $b;
});
$top = array_slice($totals, 0, 3);
foreach ($top as $cid => $t) {
    echo "customer $cid " . money($t) . "\n";
}
'''

ORDERS_FIXES = [
    ("    $o['tag'] = strtoupper(trim($o['tag']));\n}", "    $o['tag'] = strtoupper(trim($o['tag']));\n}\nunset($o);"),
    ("if (empty($o['id'])) {", "if ($o['id'] === '') {"),
    ("if (strpos($o['tag'], 'VOID')) {", "if (strpos($o['tag'], 'VOID') !== false) {"),
    ("if (in_array($o['id'], $seen)) {", "if (in_array($o['id'], $seen, true)) {"),
    ("$cents = (int)($o['amount'] * 100);", "$cents = (int)round($o['amount'] * 100);"),
    ("$top = array_slice($totals, 0, 3);", "$top = array_slice($totals, 0, 3, true);"),
]


def ref_orders(inp):
    seen = set()
    totals = {}
    count = total = 0
    for line in inp.split("\n"):
        line = line.rstrip("\r")
        if line == "":
            continue
        f = line.split(";")
        if len(f) != 4 or f[0] == "":
            continue
        if "VOID" in f[3].upper():
            continue
        if f[0] in seen:
            continue
        seen.add(f[0])
        c = _cents(f[2])
        cid = int(f[1])
        totals[cid] = totals.get(cid, 0) + c
        count += 1
        total += c
    out = ["orders=%d total=%s" % (count, _fmt_cents(total))]
    for cid in sorted(totals, key=lambda x: (-totals[x], x))[:3]:
        out.append("customer %d %s" % (cid, _fmt_cents(totals[cid])))
    return "\n".join(out) + "\n"


_VOID_TAIL = "ZZ9;1;5.00;order was void later\n"


def _orders_random(seed, n):
    r = Rng(seed)
    ids = ["A%d" % i for i in range(1, 40)] + ["7", "007", "7.0", "10", "1e1", "0", "00", "1E1", "0x1A", "26",
                                               " 7", "+7", "abc", "ABC"]
    tags = ["", "gift", "void", "VOID - duplicate", "express", "re-void", "avoid peanuts", "Void?", "ok", "n/a"]
    custs = [0, 3, 12, 40, 41, 77, 100, 2500]
    lines = []
    for _ in range(n):
        x = r.randint(1, 100)
        if x <= 4:
            lines.append("broken;line")
            continue
        if x <= 7:
            lines.append(";%d;3.00;empty id" % r.choice(custs))
            continue
        amt = "%d" % r.randint(0, 400)
        y = r.randint(1, 100)
        if y <= 60:
            amt += ".%02d" % r.randint(0, 99)
        elif y <= 75:
            amt += ".%d" % r.randint(0, 9)
        lines.append("%s;%d;%s;%s" % (r.choice(ids) if r.chance(70) else "R%d" % r.randint(1, 10 ** 6),
                                      r.choice(custs), amt, r.choice(tags)))
    lines.append("LAST%d;%d;19.99;final order" % (seed, r.choice(custs)))
    return "\n".join(lines) + "\n"


def orders_tests():
    cases = [
        ("basic", "A1;0;10.50;gift\nB2;1;3.25;\nC3;0;7;express\nD4;2;0.5;ok\nE5;1;4.75;ok\n" + _VOID_TAIL),
        ("repeated order ids and malformed lines",
         "A1;0;20.00;x\nA1;0;20.00;x\nnot a record\nB1;1;6.00;y;extra\nB2;1;6.00;y\nA1;2;1.00;z\n" + _VOID_TAIL),
        ("VOID at the start of the tag",
         "X1;0;9.00;VOID\nX2;0;5.00;void - duplicate\nX3;0;8.00;  Voided by clerk\nX4;0;2.00;ok\nX5;1;1.00;ok\n" + _VOID_TAIL),
        ("last line is a regular order", "A1;0;10.00;a\nB2;1;4.00;b\nC3;2;2.50;c\n"),
        ("two orders only", "A1;0;10.00;a\nB2;0;4.00;b\n"),
        ("order ids that look like numbers",
         "007;0;1.00;a\n7;0;2.00;b\n1e1;0;4.00;c\n10;0;8.00;d\n7.0;0;16.00;e\n00;0;32.00;f\n1E1;0;64.00;g\n"
         " 7;0;128.00;h\n" + _VOID_TAIL),
        ("amounts that are not exact in binary",
         "A1;0;19.99;a\nA2;0;0.29;b\nA3;0;1.15;c\nA4;0;4.35;d\nA5;0;0.57;e\nA6;1;1.10;f\nA7;1;8.20;g\n"
         "A8;1;2.30;h\nA9;2;0.07;i\n" + _VOID_TAIL),
        ("customer ids are kept", "A1;17;5.00;a\nA2;4;9.00;b\nA3;99;7.00;c\nA4;23;1.00;d\nA5;4;0.50;e\n"
                                  "Q7;17;1.00;was void later\n"),
        ("ties broken by smaller customer id", "A1;30;5.00;a\nA2;8;5.00;b\nA3;120;5.00;c\nA4;9;5.00;d\nA5;2;4.99;e\n"
                                             "Q7;2;1.00;was void later\n"),
        ("order id 0 is valid", "0;0;11.00;a\nB1;1;3.00;b\nB2;2;1.00;c\n" + _VOID_TAIL),
        ("CRLF line endings", "A1;0;10.50;gift\r\nB2;1;3.25;\r\nC3;2;1.00;x\r\nZ9;1;5.00;now void\r\n"),
    ]
    tests = [T(name, inp, ref_orders(inp)) for name, inp in cases]
    for i, n in enumerate([30, 200, 2500]):
        inp = _orders_random(901 + i, n)
        tests.append(T("random %d lines" % n, inp, ref_orders(inp)))
    return tests


task(id="php_bugfix_orders", lang="php", category="bugfix",
     title="Order summary with hidden bugs",
     spec=ORDERS_SPEC, given=ORDERS_GIVEN, fixes=ORDERS_FIXES, tests=orders_tests)


# ---- Java: sensor readings -------------------------------------------------

SENSORS_SPEC = r"""Input: line 1 holds N (1 <= N <= 200000). Each of the next N lines is either
  READ sensor x y value unit duration
  SAME sensorA sensorB
READ is a reading of the sensor (a name without spaces) at grid point (x, y), both integers. Only the first reading of each (sensor, x, y) point is accepted; later readings of the same point are ignored completely. value is an integer expressed in the given unit, mV or V (1 V = 1000 mV); converted to mV it lies within -2*10^9 .. 2*10^9. duration is an integer from 1 to 10^6.

Output:
1. For every sensor with at least one accepted reading, in ascending order of name (String natural order):
   <sensor> n=<number of accepted readings> energy=<sum over its accepted readings of value_in_mV * duration>
2. One line starting with  smallest:  followed by the (up to) three smallest accepted values in mV, in ascending order, each preceded by a space (just  smallest:  when there is no accepted reading).
3. For every SAME line, in input order:  yes  when both sensors have the same number of accepted readings (a sensor without accepted readings has 0), otherwise  no .
All numbers must be exact."""

SENSORS_GIVEN = r'''import java.util.*;

public class Main {
    static class Point {
        final String sensor;
        final int x, y;

        Point(String sensor, int x, int y) {
            this.sensor = sensor;
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Point)) return false;
            Point p = (Point) o;
            return sensor.equals(p.sensor) && x == p.x && y == p.y;
        }
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = Integer.parseInt(in.nextLine().trim());
        Set<Point> seen = new HashSet<>();
        Map<String, Integer> counts = new TreeMap<>();
        Map<String, Long> energy = new TreeMap<>();
        List<Integer> values = new ArrayList<>();
        List<String[]> queries = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            String[] f = in.nextLine().trim().split("\\s+");
            if (f[0].equals("SAME")) {
                queries.add(f);
                continue;
            }
            String sensor = f[1];
            int x = Integer.parseInt(f[2]);
            int y = Integer.parseInt(f[3]);
            int value = Integer.parseInt(f[4]);
            String unit = f[5];
            int duration = Integer.parseInt(f[6]);
            if (!seen.add(new Point(sensor, x, y))) continue;
            if (unit == "V") value = value * 1000;
            counts.put(sensor, counts.getOrDefault(sensor, 0) + 1);
            long e = value * duration;
            energy.put(sensor, energy.getOrDefault(sensor, 0L) + e);
            values.add(value);
        }

        for (String s : counts.keySet()) {
            System.out.println(s + " n=" + counts.get(s) + " energy=" + energy.get(s));
        }

        Collections.sort(values, (a, b) -> a - b);
        StringBuilder sb = new StringBuilder("smallest:");
        for (int i = 0; i < 3 && i < values.size(); i++) sb.append(' ').append(values.get(i));
        System.out.println(sb);

        for (String[] q : queries) {
            Integer a = counts.get(q[1]);
            Integer b = counts.get(q[2]);
            System.out.println(a == b ? "yes" : "no");
        }
    }
}
'''

SENSORS_FIXES = [
    ("            return sensor.equals(p.sensor) && x == p.x && y == p.y;\n        }",
     "            return sensor.equals(p.sensor) && x == p.x && y == p.y;\n        }\n\n"
     "        @Override\n        public int hashCode() {\n            return Objects.hash(sensor, x, y);\n        }"),
    ('if (unit == "V")', 'if (unit.equals("V"))'),
    ("long e = value * duration;", "long e = (long) value * duration;"),
    ("(a, b) -> a - b", "(a, b) -> Integer.compare(a, b)"),
    ("            Integer a = counts.get(q[1]);\n            Integer b = counts.get(q[2]);",
     "            int a = counts.getOrDefault(q[1], 0);\n            int b = counts.getOrDefault(q[2], 0);"),
]


def ref_sensors(inp):
    lines = inp.split("\n")
    n = int(lines[0])
    seen = set()
    counts, energy, values, queries = {}, {}, [], []
    for ln in lines[1:1 + n]:
        f = ln.split()
        if f[0] == "SAME":
            queries.append((f[1], f[2]))
            continue
        key = (f[1], int(f[2]), int(f[3]))
        if key in seen:
            continue
        seen.add(key)
        v = int(f[4]) * (1000 if f[5] == "V" else 1)
        counts[f[1]] = counts.get(f[1], 0) + 1
        energy[f[1]] = energy.get(f[1], 0) + v * int(f[6])
        values.append(v)
    out = ["%s n=%d energy=%d" % (s, counts[s], energy[s]) for s in sorted(counts)]
    out.append("smallest:" + "".join(" %d" % v for v in sorted(values)[:3]))
    for a, b in queries:
        out.append("yes" if counts.get(a, 0) == counts.get(b, 0) else "no")
    return "\n".join(out) + "\n"


def _sensors_input(lines):
    return "%d\n%s\n" % (len(lines), "\n".join(lines))


def _sensors_random(seed, n, extremes):
    r = Rng(seed)
    names = ["alpha", "beta", "gamma", "delta", "Zeta", "eta9"]
    lines = []
    for _ in range(n):
        if r.chance(8):
            lines.append("SAME %s %s" % (r.choice(names + ["ghost"]), r.choice(names + ["ghost", "phantom"])))
            continue
        unit = "V" if r.chance(40) else "mV"
        lim = 2 * 10 ** 9 if extremes and r.chance(30) else 5000
        v = r.randint(-lim, lim)
        if unit == "V":
            v = r.randint(-min(lim, 2 * 10 ** 6), min(lim, 2 * 10 ** 6))
        lines.append("READ %s %d %d %d %s %d" % (r.choice(names), r.randint(-6, 6), r.randint(-6, 6), v, unit,
                                                 r.randint(1, 10 ** 6 if extremes else 50)))
    # Two sensors with exactly the same (large) number of accepted readings.
    for s in ("twinA", "twinB"):
        for i in range(160):
            lines.append("READ %s %d %d %d mV 2" % (s, i, -i, r.randint(-99, 99)))
    r.shuffle(lines)
    lines += ["SAME twinA twinB", "SAME twinB alpha", "SAME ghost phantom", "SAME twinA ghost"]
    return _sensors_input(lines)


def sensors_tests():
    def rd(s, x, y, v, unit="mV", d=1):
        return "READ %s %d %d %d %s %d" % (s, x, y, v, unit, d)

    big = [rd("a", i, 0, 10 + i) for i in range(200)] + [rd("b", 0, i, 500 + i) for i in range(200)] + \
          [rd("c", i, i, 7) for i in range(199)] + ["SAME a b", "SAME a c", "SAME b a"]
    cached = [rd("a", i, 0, 10 + i) for i in range(127)] + [rd("b", 0, i, 500 + i) for i in range(127)] + ["SAME a b"]
    cases = [
        ("basic", _sensors_input([rd("s2", 0, 0, 30, d=2), rd("s1", 1, 0, 10, d=3), rd("s1", 0, 1, -5),
                                  rd("s2", 5, 5, 8), "SAME s1 s2", rd("s3", 0, 0, 1), "SAME s1 s3",
                                  "SAME nobody ghost", "SAME s3 nobody"])),
        ("fewer than three values", _sensors_input([rd("k", 1, 1, 42, d=10), "SAME k k"])),
        ("no readings at all", _sensors_input(["SAME a b"])),
        ("repeated points are ignored", _sensors_input([rd("s1", 1, 1, 10), rd("s1", 1, 1, 999, d=9), rd("s1", 1, 2, 20),
                                                        rd("s2", 1, 1, 30), rd("s1", 1, 2, -50), rd("s2", 1, 1, 30),
                                                        "SAME s1 s2"])),
        ("many repeated points", _sensors_input([rd("grid", i % 5, i % 3, i) for i in range(90)] + ["SAME grid grid"])),
        ("volt readings are converted", _sensors_input([rd("s1", 0, 0, 3, "V", 2), rd("s1", 0, 1, 2500, "mV", 2),
                                                        rd("s2", 0, 0, -1, "V"), rd("s2", 9, 9, 7, "V", 5)])),
        ("energy above 32 bits", _sensors_input([rd("s1", 0, 0, 2000000, "mV", 1000000), rd("s1", 0, 1, -1999999, "mV", 999999),
                                                 rd("s2", 0, 0, 70000, "mV", 40000)])),
        ("extreme values are ordered correctly", _sensors_input([rd("s1", 0, 0, 2000000000), rd("s1", 0, 1, -2000000000),
                                                                rd("s1", 0, 2, 5), rd("s1", 0, 3, 1500000000),
                                                                rd("s1", 0, 4, -1900000000)])),
        ("extreme values, two readings", _sensors_input([rd("s1", 0, 0, -2000000000), rd("s1", 0, 1, 2000000000)])),
        ("equal counts up to 127", _sensors_input(cached)),
        ("equal counts above 127", _sensors_input(big)),
    ]
    tests = [T(name, inp, ref_sensors(inp)) for name, inp in cases]
    for i, (n, ext) in enumerate([(60, False), (500, True), (20000, True)]):
        inp = _sensors_random(1001 + i, n, ext)
        tests.append(T("random %d lines" % n, inp, ref_sensors(inp)))
    return tests


task(id="java_bugfix_sensors", lang="java", category="bugfix",
     title="Sensor report with hidden bugs",
     spec=SENSORS_SPEC, given=SENSORS_GIVEN, fixes=SENSORS_FIXES, tests=sensors_tests)


# ==========================================================================
# OPTIMIZATION TASKS (one per language)
#
# The given program is correct but slow. Small tests check that behaviour is
# preserved; "tier" tests are large inputs with a strict time limit. The
# limits sit far away from both the slow and the fast version (>= 4x margin on
# the reference machine), so they do not depend on small speed differences.
# ==========================================================================

# ---- Python: submatrices with a given sum ---------------------------------

SUBSUM_SPEC = r"""Input: line 1 holds r, c and k (1 <= r, c <= 3000, r <= 25 or c <= 25, |k| <= 10^9). Then r lines of c integers each: the rows of a matrix A (|A[i][j]| <= 1000).
Output: the number of submatrices of A whose elements sum to exactly k. A submatrix is a non-empty block of consecutive rows and consecutive columns."""

SUBSUM_GIVEN = r'''import sys


def main():
    data = sys.stdin.read().split()
    r, c, k = int(data[0]), int(data[1]), int(data[2])
    a = []
    pos = 3
    for _ in range(r):
        a.append([int(x) for x in data[pos:pos + c]])
        pos += c
    count = 0
    for r1 in range(r):
        for r2 in range(r1, r):
            for c1 in range(c):
                for c2 in range(c1, c):
                    s = 0
                    for i in range(r1, r2 + 1):
                        for j in range(c1, c2 + 1):
                            s += a[i][j]
                    if s == k:
                        count += 1
    print(count)


main()
'''

SUBSUM_FAST = r'''import sys
from itertools import accumulate


def main():
    data = sys.stdin.buffer.read().split()
    r, c, k = int(data[0]), int(data[1]), int(data[2])
    vals = list(map(int, data[3:3 + r * c]))
    rows = [vals[i * c:(i + 1) * c] for i in range(r)]
    if r > c:
        rows = [list(col) for col in zip(*rows)]
        r, c = c, r
    count = 0
    for top in range(r):
        col = [0] * c
        for bottom in range(top, r):
            col = [x + y for x, y in zip(col, rows[bottom])]
            seen = {0: 1}
            get = seen.get
            for s in accumulate(col):
                count += get(s - k, 0)
                seen[s] = get(s, 0) + 1
    print(count)


main()
'''


def ref_subsum(k, rows):
    if len(rows) > len(rows[0]):
        rows = [list(col) for col in zip(*rows)]
    count = 0
    for top in range(len(rows)):
        col = [0] * len(rows[0])
        for row in rows[top:]:
            col = [x + y for x, y in zip(col, row)]
            seen = {0: 1}
            s = 0
            for x in col:
                s += x
                count += seen.get(s - k, 0)
                seen[s] = seen.get(s, 0) + 1
    return "%d\n" % count


def _subsum_case(name, k, rows, **kw):
    inp = "%d %d %d\n%s\n" % (len(rows), len(rows[0]), k,
                              "\n".join(" ".join(map(str, row)) for row in rows))
    return T(name, inp, ref_subsum(k, rows), **kw)


def _subsum_matrix(r, c, m, salt):
    """Small values in [-m//2, m//2] from a closed form, so that many submatrices match."""
    return [[(i * i * 31 + j * j * 7 + i * j * 13 + j * 5 + i * 3 + salt) % m - m // 2
             for j in range(c)] for i in range(r)]


def subsum_tests():
    rnd = Rng(1100)

    def rand(r, c, lo, hi):
        return [[rnd.randint(lo, hi) for _ in range(c)] for _ in range(r)]

    tests = [
        _subsum_case("small example", 4, [[1, 2, -1, 3], [0, 2, 2, -2], [3, -3, 1, 1]], weight=0.5),
        _subsum_case("all zeros, k = 0", 0, [[0] * 9 for _ in range(7)], weight=0.5),
        _subsum_case("single cell", -5, [[-5]], weight=0.5),
        _subsum_case("single row", 3, rand(1, 60, -4, 4), weight=0.5),
        _subsum_case("single column", -2, rand(60, 1, -4, 4), weight=0.5),
        _subsum_case("no match", 10 ** 9, rand(9, 11, -1000, 1000), weight=0.5),
        _subsum_case("more rows than columns", 6, rand(14, 5, -5, 5), weight=0.5),
    ]
    # Tier 1 rejects only the given program. Tiers 2 and 3 need O(min(r,c)^2 * max(r,c)):
    # pairing the rows (or the columns) in both orientations takes about 30 s on one of them.
    tests.append(_subsum_case("tier 1: 25 x 150", 3, _subsum_matrix(25, 150, 7, 1),
                              limit=3.0, weight=2.0, timed=True))
    tests.append(_subsum_case("tier 2: 25 x 3000 (wide)", 7, _subsum_matrix(25, 3000, 9, 2),
                              limit=3.0, weight=3.0, timed=True))
    tests.append(_subsum_case("tier 3: 3000 x 25 (tall)", -4, _subsum_matrix(3000, 25, 9, 3),
                              limit=3.0, weight=3.0, timed=True))
    return tests


task(id="py_opt_subsum", lang="python", category="optimize",
     title="Count submatrices with a given sum",
     spec=SUBSUM_SPEC, given=SUBSUM_GIVEN, solution=SUBSUM_FAST, tests=subsum_tests)


# ---- C: inversion count ----------------------------------------------------

INV_SPEC = r"""Input: line 1 holds n (1 <= n <= 2000000). Then n integers a_1..a_n (|a_i| <= 10^9) separated by whitespace.
Output: the number of pairs (i, j) with i < j and a_i > a_j."""

INV_GIVEN = r'''#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    int *a = malloc(sizeof(int) * (size_t)n);
    if (!a) return 1;
    for (int i = 0; i < n; i++) {
        if (scanf("%d", &a[i]) != 1) return 1;
    }
    long long inversions = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (a[i] > a[j]) inversions++;
        }
    }
    printf("%lld\n", inversions);
    free(a);
    return 0;
}
'''

INV_FAST = r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long sort_count(int *a, int *tmp, int lo, int hi) {
    if (hi - lo < 2) return 0;
    int mid = lo + (hi - lo) / 2;
    long long inv = sort_count(a, tmp, lo, mid) + sort_count(a, tmp, mid, hi);
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else { tmp[k++] = a[j++]; inv += mid - i; }
    }
    while (i < mid) tmp[k++] = a[i++];
    while (j < hi) tmp[k++] = a[j++];
    memcpy(a + lo, tmp + lo, sizeof(int) * (size_t)(hi - lo));
    return inv;
}

int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    int *a = malloc(sizeof(int) * (size_t)n), *tmp = malloc(sizeof(int) * (size_t)n);
    if (!a || !tmp) return 1;
    for (int i = 0; i < n; i++) if (scanf("%d", &a[i]) != 1) return 1;
    printf("%lld\n", sort_count(a, tmp, 0, n));
    free(a);
    free(tmp);
    return 0;
}
'''


def _inv_small(name, a):
    inv = sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j])
    return T(name, "%d\n%s\n" % (len(a), " ".join(map(str, a))), "%d\n" % inv, weight=0.5)


def _inv_blocks(seed, n, nblocks):
    """Large input whose inversion count is known in closed form."""
    r = Rng(seed)
    cuts = sorted(set(r.randint(1, n - 1) for _ in range(nblocks - 1)))
    bounds = [0] + cuts + [n]
    sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    ranks = list(range(len(sizes)))
    r.shuffle(ranks)
    width = n + 1
    offset = -(len(sizes) * width) // 2
    chunks = []
    inv = 0
    for size, rank in zip(sizes, ranks):
        base = rank * width + offset
        kind = r.randint(0, 2)
        if kind == 0:                                   # strictly descending run
            chunks.append(" ".join(map(str, range(base + size - 1, base - 1, -1))))
            inv += size * (size - 1) // 2
        elif kind == 1:                                 # ascending run
            chunks.append(" ".join(map(str, range(base, base + size))))
        else:                                           # constant run
            chunks.append(" ".join([str(base)] * size))
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            if ranks[i] > ranks[j]:
                inv += sizes[i] * sizes[j]
    return "%d\n%s\n" % (n, "\n".join(chunks)), "%d\n" % inv


def inv_tests():
    r = Rng(1200)
    tests = [
        _inv_small("small example", [3, 1, 2, 2, -7, 10, 0]),
        _inv_small("single element", [42]),
        _inv_small("sorted and reversed", list(range(50)) + list(range(50, 0, -1))),
        _inv_small("all equal", [5] * 80),
        _inv_small("random with duplicates and extremes",
                   [r.choice([-10 ** 9, 10 ** 9, r.randint(-20, 20), r.randint(-10 ** 9, 10 ** 9)])
                    for _ in range(500)]),
    ]
    for label, seed, n, blocks, w in (("tier 1: n = 400000", 1201, 400000, 60, 2.5),
                                      ("tier 2: n = 1000000", 1202, 1000000, 150, 2.5),
                                      ("tier 3: n = 2000000", 1203, 2000000, 300, 2.5)):
        inp, exp = _inv_blocks(seed, n, blocks)
        tests.append(T(label, inp, exp, limit=2.0, weight=w, timed=True))
    return tests


task(id="c_opt_inversions", lang="c", category="optimize",
     title="Count inversions",
     spec=INV_SPEC, given=INV_GIVEN, solution=INV_FAST, tests=inv_tests)


# ---- C#: anagram lookups ---------------------------------------------------

ANAGRAM_SPEC = r"""Input: line 1 holds N and Q (1 <= N, Q <= 200000). The next N lines are the dictionary: one word per line, lowercase letters a-z, length 1 to 12 (the same word may appear more than once; every occurrence counts). The next Q lines are query words in the same format.
Output: for each query, in order, one line with the number of dictionary entries that are anagrams of the query (same letters with the same multiplicities; an identical word is an anagram too)."""

ANAGRAM_GIVEN = r'''using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static bool IsAnagram(string a, string b)
    {
        if (a.Length != b.Length) return false;
        string x = new string(a.OrderBy(c => c).ToArray());
        string y = new string(b.OrderBy(c => c).ToArray());
        return x == y;
    }

    static void Main()
    {
        string[] first = Console.ReadLine().Split(' ');
        int n = int.Parse(first[0]);
        int q = int.Parse(first[1]);
        var dict = new List<string>();
        for (int i = 0; i < n; i++) dict.Add(Console.ReadLine().Trim());

        string output = "";
        for (int i = 0; i < q; i++)
        {
            string w = Console.ReadLine().Trim();
            int count = 0;
            foreach (string d in dict)
            {
                if (IsAnagram(w, d)) count++;
            }
            output += count + "\n";
        }
        Console.Write(output);
    }
}
'''

ANAGRAM_FAST = r'''using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

class Program
{
    static string Key(string w)
    {
        char[] c = w.ToCharArray();
        Array.Sort(c);
        return new string(c);
    }

    static void Main()
    {
        var input = new StreamReader(Console.OpenStandardInput());
        string[] first = input.ReadLine().Split(' ');
        int n = int.Parse(first[0]);
        int q = int.Parse(first[1]);
        var counts = new Dictionary<string, int>();
        for (int i = 0; i < n; i++)
        {
            string k = Key(input.ReadLine().Trim());
            int v;
            counts.TryGetValue(k, out v);
            counts[k] = v + 1;
        }
        var sb = new StringBuilder();
        for (int i = 0; i < q; i++)
        {
            int v;
            counts.TryGetValue(Key(input.ReadLine().Trim()), out v);
            sb.Append(v).Append('\n');
        }
        Console.Write(sb.ToString());
    }
}
'''


def ref_anagram(words, queries):
    counts = {}
    for w in words:
        k = "".join(sorted(w))
        counts[k] = counts.get(k, 0) + 1
    return "".join("%d\n" % counts.get("".join(sorted(w)), 0) for w in queries)


def _anagram_case(name, words, queries, **kw):
    inp = "%d %d\n%s\n%s\n" % (len(words), len(queries), "\n".join(words), "\n".join(queries))
    return T(name, inp, ref_anagram(words, queries), **kw)


def _anagram_big(seed, n, q, length=8):
    r = Rng(seed)
    pool = ["".join(chr(97 + r.randint(0, 7)) for _ in range(length)) for _ in range(4000)]
    npool = len(pool)

    def pick():
        w = pool[r._step() % npool]
        k = r._step() % length
        return w[k:] + w[:k]

    return [pick() for _ in range(n)], [pick() for _ in range(q)]


def anagram_tests():
    r = Rng(1300)
    letters = "abc"
    small_words = ["".join(r.choice(letters) for _ in range(r.randint(1, 5))) for _ in range(250)]
    small_q = ["".join(r.choice(letters) for _ in range(r.randint(1, 5))) for _ in range(250)]
    tests = [
        _anagram_case("small example", ["listen", "silent", "enlist", "google", "tinsel", "listen", "a", "ab", "ba"],
                      ["inlets", "gogole", "a", "b", "ab", "listenn", "silent"], weight=0.5),
        _anagram_case("no matches", ["abc", "abd"], ["abcc", "ab", "x"], weight=0.5),
        _anagram_case("same letters, different counts", ["aab", "abb", "aba", "baa", "aabb"],
                      ["aab", "abb", "ab", "bbaa"], weight=0.5),
        _anagram_case("twelve-letter words", ["abcdefghijkl", "lkjihgfedcba", "abcdefghijkk", "zzzzzzzzzzzz"],
                      ["abcdefghijkl", "zzzzzzzzzzzz", "kkjihgfedcba", "abcdefghijk"], weight=0.5),
        _anagram_case("random small", small_words, small_q, weight=0.5),
    ]
    for label, seed, n, q, w in (("tier 1: N = Q = 5000", 1301, 5000, 5000, 2.0),
                                 ("tier 2: N = 200000, Q = 20000", 1302, 200000, 20000, 2.5),
                                 ("tier 3: N = Q = 200000", 1303, 200000, 200000, 3.0)):
        words, queries = _anagram_big(seed, n, q)
        tests.append(_anagram_case(label, words, queries, limit=2.0, weight=w, timed=True))
    return tests


task(id="cs_opt_anagrams", lang="csharp", category="optimize",
     title="Anagram lookups",
     spec=ANAGRAM_SPEC, given=ANAGRAM_GIVEN, solution=ANAGRAM_FAST, tests=anagram_tests)


# ---- PHP: shortest path in a grid -----------------------------------------

MAZE_SPEC = r"""Input: line 1 holds H and W (1 <= H, W <= 1000). The next H lines hold W characters each: '.' is a free cell, '#' a wall, 'S' the start and 'E' the end (exactly one S and one E).
Output: the minimum number of moves needed to go from S to E moving one cell up, down, left or right through cells that are not walls, or -1 when E cannot be reached."""

MAZE_GIVEN = r'''<?php
$first = explode(' ', trim(fgets(STDIN)));
$h = (int)$first[0];
$w = (int)$first[1];
$grid = [];
for ($i = 0; $i < $h; $i++) {
    $grid[] = rtrim(fgets(STDIN), "\r\n");
}
$start = null;
$end = null;
for ($r = 0; $r < $h; $r++) {
    for ($c = 0; $c < $w; $c++) {
        if ($grid[$r][$c] === 'S') {
            $start = [$r, $c];
        } elseif ($grid[$r][$c] === 'E') {
            $end = [$r, $c];
        }
    }
}

$visited = [$start[0] . ',' . $start[1]];
$queue = [[$start[0], $start[1], 0]];
$answer = -1;
while (count($queue) > 0) {
    $cur = array_shift($queue);
    if ($cur[0] == $end[0] && $cur[1] == $end[1]) {
        $answer = $cur[2];
        break;
    }
    foreach ([[1, 0], [-1, 0], [0, 1], [0, -1]] as $d) {
        $nr = $cur[0] + $d[0];
        $nc = $cur[1] + $d[1];
        if ($nr < 0 || $nc < 0 || $nr >= $h || $nc >= $w) {
            continue;
        }
        if ($grid[$nr][$nc] === '#') {
            continue;
        }
        $key = $nr . ',' . $nc;
        if (in_array($key, $visited)) {
            continue;
        }
        $visited[] = $key;
        $queue[] = [$nr, $nc, $cur[2] + 1];
    }
}
echo $answer, "\n";
'''

MAZE_FAST = r'''<?php
$first = explode(' ', trim(fgets(STDIN)));
$h = (int)$first[0];
$w = (int)$first[1];
$cells = '';
for ($i = 0; $i < $h; $i++) {
    $cells .= rtrim(fgets(STDIN), "\r\n");
}
$start = strpos($cells, 'S');
$end = strpos($cells, 'E');
$dist = array_fill(0, $h * $w, -1);
$dist[$start] = 0;
$queue = [$start];
$head = 0;
$answer = -1;
while ($head < count($queue)) {
    $cur = $queue[$head++];
    if ($cur === $end) {
        $answer = $dist[$cur];
        break;
    }
    $d = $dist[$cur] + 1;
    $col = $cur % $w;
    if ($cur >= $w) {
        $n = $cur - $w;
        if ($dist[$n] < 0 && $cells[$n] !== '#') { $dist[$n] = $d; $queue[] = $n; }
    }
    if ($cur + $w < $h * $w) {
        $n = $cur + $w;
        if ($dist[$n] < 0 && $cells[$n] !== '#') { $dist[$n] = $d; $queue[] = $n; }
    }
    if ($col > 0) {
        $n = $cur - 1;
        if ($dist[$n] < 0 && $cells[$n] !== '#') { $dist[$n] = $d; $queue[] = $n; }
    }
    if ($col < $w - 1) {
        $n = $cur + 1;
        if ($dist[$n] < 0 && $cells[$n] !== '#') { $dist[$n] = $d; $queue[] = $n; }
    }
}
echo $answer, "\n";
'''


def ref_maze(rows):
    h, w = len(rows), len(rows[0])
    cells = "".join(rows)
    start, end = cells.index("S"), cells.index("E")
    dist = [-1] * (h * w)
    dist[start] = 0
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == end:
            return "%d\n" % dist[cur]
        d = dist[cur] + 1
        col = cur % w
        for ok, nxt in ((cur >= w, cur - w), (cur + w < h * w, cur + w), (col > 0, cur - 1),
                        (col < w - 1, cur + 1)):
            if ok and dist[nxt] < 0 and cells[nxt] != "#":
                dist[nxt] = d
                queue.append(nxt)
    return "-1\n"


def _maze_text(rows):
    return "%d %d\n%s\n" % (len(rows), len(rows[0]), "\n".join(rows))


def _maze_random(seed, h, w, wall_pct):
    r = Rng(seed)
    rows = [["#" if r.chance(wall_pct) else "." for _ in range(w)] for _ in range(h)]
    rows[0][0] = "S"
    rows[h - 1][w - 1] = "E"
    return ["".join(row) for row in rows]


def _maze_serpentine(seed, h, w):
    """Open rows alternate with wall rows that have a single gap; the answer is a closed form."""
    r = Rng(seed)
    rows = []
    col = 0
    moves = 0
    for i in range(h):
        if i % 2 == 0:
            rows.append(["."] * w)
        else:
            gap = r.randint(0, w - 1)
            row = ["#"] * w
            row[gap] = "."
            rows.append(row)
            moves += abs(gap - col) + 2
            col = gap
    moves += abs((w - 1) - col)
    rows[0][0] = "S"
    rows[h - 1][w - 1] = "E"
    return ["".join(row) for row in rows], "%d\n" % moves


def maze_tests():
    def small(name, rows):
        return T(name, _maze_text(rows), ref_maze(rows), weight=0.5)

    tests = [
        small("small example", ["S..#....", ".#.#.##.", ".#...#..", ".####.#.", "......#E"]),
        small("unreachable", ["S.#.", "..#.", "###.", "...E"]),
        small("adjacent", ["SE"]),
        small("single column", ["S", ".", ".", "E"]),
        small("random 20x20", _maze_random(1401, 20, 20, 22)),
    ]
    rows = _maze_random(1402, 250, 250, 18)
    tests.append(T("tier 1: 250 x 250", _maze_text(rows), ref_maze(rows), limit=3.0, weight=1.5, timed=True))
    rows = _maze_random(1403, 400, 400, 15)
    tests.append(T("tier 2: 400 x 400", _maze_text(rows), ref_maze(rows), limit=3.0, weight=2.0, timed=True))
    rows = ["." * 1000 for _ in range(1000)]
    rows[0] = "S" + rows[0][1:]
    rows[-1] = rows[-1][:-1] + "E"
    tests.append(T("tier 3: 1000 x 1000 open field", _maze_text(rows), "1998\n", limit=3.0, weight=2.0, timed=True))
    rows, exp = _maze_serpentine(1404, 999, 1000)
    tests.append(T("tier 4: 999 x 1000 corridors", _maze_text(rows), exp, limit=3.0, weight=2.0, timed=True))
    return tests


task(id="php_opt_maze", lang="php", category="optimize",
     title="Shortest path in a grid",
     spec=MAZE_SPEC, given=MAZE_GIVEN, solution=MAZE_FAST, tests=maze_tests)


# ---- Java: longest increasing subsequence ---------------------------------

LIS_SPEC = r"""Input: line 1 holds n (1 <= n <= 1500000). Then n integers a_1..a_n (|a_i| <= 10^9) separated by whitespace.
Output: the length of the longest strictly increasing subsequence of a."""

LIS_GIVEN = r'''import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int[] a = new int[n];
        for (int i = 0; i < n; i++) a[i] = in.nextInt();

        int[] best = new int[n];
        int answer = 0;
        for (int i = 0; i < n; i++) {
            best[i] = 1;
            for (int j = 0; j < i; j++) {
                if (a[j] < a[i] && best[j] + 1 > best[i]) best[i] = best[j] + 1;
            }
            answer = Math.max(answer, best[i]);
        }
        System.out.println(answer);
    }
}
'''

LIS_FAST = r'''import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        DataInputStream in = new DataInputStream(new BufferedInputStream(System.in, 1 << 16));
        int n = nextInt(in);
        int[] tails = new int[n];
        int len = 0;
        for (int i = 0; i < n; i++) {
            int v = nextInt(in);
            int lo = 0, hi = len;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (tails[mid] < v) lo = mid + 1; else hi = mid;
            }
            tails[lo] = v;
            if (lo == len) len++;
        }
        System.out.println(len);
    }

    private static int nextInt(DataInputStream in) throws IOException {
        int ret = 0;
        int b = in.read();
        while (b != '-' && (b < '0' || b > '9')) b = in.read();
        boolean neg = false;
        if (b == '-') { neg = true; b = in.read(); }
        while (b >= '0' && b <= '9') { ret = ret * 10 + (b - '0'); b = in.read(); }
        return neg ? -ret : ret;
    }
}
'''


def ref_lis(a):
    import bisect
    tails = []
    bl = bisect.bisect_left
    for v in a:
        i = bl(tails, v)
        if i == len(tails):
            tails.append(v)
        else:
            tails[i] = v
    return "%d\n" % len(tails)


def _lis_case(name, a, **kw):
    return T(name, "%d\n%s\n" % (len(a), " ".join(map(str, a))), ref_lis(a), **kw)


def lis_tests():
    r = Rng(1500)
    tests = [
        _lis_case("small example", [10, 9, 2, 5, 3, 7, 101, 18, 4, 20], weight=0.5),
        _lis_case("duplicates are not increasing", [5, 5, 5, 5, 6, 6, 7, 7, 7], weight=0.5),
        _lis_case("single element", [-3], weight=0.5),
        _lis_case("decreasing with extremes", [10 ** 9, 5, 0, -5, -10 ** 9], weight=0.5),
        _lis_case("random small", [r.randint(-300, 300) for _ in range(800)], weight=0.5),
    ]
    a = [(i * 7919 + 13) % 1000003 - 500000 for i in range(200000)]
    tests.append(_lis_case("tier 1: n = 200000", a, limit=2.0, weight=2.5, timed=True))
    a = [(i * 104729 + 7) % 999983 + (i // 3) for i in range(600000)]
    tests.append(_lis_case("tier 2: n = 600000", a, limit=2.0, weight=2.5, timed=True))
    a = [((i * 15485863 + 11) % 1000000007) - 500000000 if i % 5 else i * 300 - 400000000 for i in range(1500000)]
    tests.append(_lis_case("tier 3: n = 1500000", a, limit=3.0, weight=2.5, timed=True))
    return tests


task(id="java_opt_lis", lang="java", category="optimize",
     title="Longest increasing subsequence",
     spec=LIS_SPEC, given=LIS_GIVEN, solution=LIS_FAST, tests=lis_tests)


# ==========================================================================
# SQL TASKS (SQLite, in-memory, through Python's sqlite3 module)
# ==========================================================================

def sql_statements(script):
    """Split a script into statements (understands triggers, strings and comments)."""
    out, buf = [], ""
    for ch in script:
        buf += ch
        if ch == ";" and sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    tail = re.sub(r"--[^\n]*|/\*.*?\*/", "", buf, flags=re.S).strip()
    if tail:
        out.append(buf.strip())
    cleaned = []
    for s in out:
        if re.sub(r"--[^\n]*|/\*.*?\*/", "", s, flags=re.S).strip(" \t\r\n;"):
            cleaned.append(s)
    return cleaned


def sql_head(stmt):
    """First keyword(s) of a statement, upper-cased, comments removed."""
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", stmt, flags=re.S).strip().upper()
    return re.sub(r"\s+", " ", s)[:40]


def sql_connect():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def sql_guard(conn, seconds):
    """Abort any statement still running after `seconds`."""
    end = time.monotonic() + seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > end else 0, 20000)


def norm_rows(rows):
    out = []
    for row in rows:
        vals = []
        for v in row:
            if isinstance(v, float):
                v = int(v) if v == int(v) else round(v, 6)
            elif isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            vals.append(v)
        out.append(tuple(vals))
    return out


def sql_fetch(conn, sql, cap=200000):
    cur = conn.execute(sql)
    rows = cur.fetchmany(cap)
    return norm_rows(rows)


def R(name, passed, detail="", weight=1.0):
    return {"name": name, "passed": bool(passed), "detail": detail, "weight": weight, "secs": 0.0}


def run_script(conn, statements):
    """Execute a script the way the sqlite3 shell does: a statement that fails is skipped and
    the rest still runs, so one invalid trigger does not void the tables next to it.
    Returns the first error message, or "" when every statement ran.
    """
    first = ""
    for s in statements:
        try:
            conn.execute(s)
        except Exception as e:
            if not first:
                first = shorten("%s (in: %s)" % (e, " ".join(s.split())), 200)
    return first


def run_steps(schema, steps):
    """Run a behaviour check on a fresh database. Returns (passed, detail)."""
    conn = sql_connect()
    sql_guard(conn, 10)
    try:
        broken = run_script(conn, schema)
        ok, detail = check_steps(conn, steps)
        if not ok and broken:
            detail = "%s | script error: %s" % (detail, broken)
        return ok, detail
    finally:
        conn.close()


def check_steps(conn, steps):
    for step in steps:
        kind, sql = step[0], step[1]
        if kind == "do":
            try:
                conn.execute(sql)
            except Exception as e:
                return False, "valid statement was rejected: %s -> %s" % (shorten(sql, 120), shorten(str(e), 120))
        elif kind == "fail":
            try:
                conn.execute(sql)
            except Exception as e:
                # A table or column missing from a broken script is not a rule being enforced.
                if str(e).startswith("no such "):
                    return False, "statement failed for the wrong reason: %s -> %s" % (
                        shorten(sql, 120), shorten(str(e), 120))
                continue
            return False, "statement should have failed: %s" % shorten(sql, 160)
        else:
            try:
                got = sql_fetch(conn, sql)
            except Exception as e:
                return False, "query failed: %s -> %s" % (shorten(sql, 120), shorten(str(e), 120))
            want = norm_rows(step[2])
            if got != want:
                return False, "%s -> got %s, expected %s" % (shorten(sql, 100), shorten(repr(got), 160),
                                                          shorten(repr(want), 160))
    return True, ""


def grade_behaviour(answer, checks):
    schema = sql_statements(answer)
    if not schema:
        return [R(name, False, "no SQL found") for name, _ in checks]
    return [R(name, *run_steps(schema, steps)) for name, steps in checks]


def grade_query(answer, datasets, limit=10.0):
    """Query tasks: the answer is ONE SELECT; it is run against several hidden datasets."""
    stmts = sql_statements(answer)
    results = []
    bad = None
    if not stmts:
        bad = "no SQL found"
    else:
        query = stmts[-1]
        if not (sql_head(query).startswith("SELECT") or sql_head(query).startswith("WITH")):
            bad = "the answer is not a SELECT statement"
    for name, setup, expected in datasets:
        if bad:
            results.append(R(name, False, bad))
            continue
        conn = sql_connect()
        try:
            for s in setup:
                if isinstance(s, tuple):
                    conn.executemany(s[0], s[1])
                else:
                    conn.execute(s)
            sql_guard(conn, limit)
            t0 = time.monotonic()
            try:
                got = sql_fetch(conn, query)
            except Exception as e:
                results.append(R(name, False, "query failed: %s" % shorten(str(e), 200)))
                continue
            want = norm_rows(expected)
            res = R(name, got == want, "" if got == want else
                    "got %d rows %s, expected %d rows %s" % (len(got), shorten(repr(got[:4]), 200),
                                                          len(want), shorten(repr(want[:4]), 200)))
            res["secs"] = time.monotonic() - t0
            results.append(res)
        finally:
            conn.close()
    return results


# ---- SQL 1: design a schema with constraints, triggers and views ----------

SHOP_SPEC = r"""Design a SQLite schema for a small shop.
Answer with ONE ```sql block containing only DDL statements (CREATE TABLE / CREATE INDEX / CREATE TRIGGER / CREATE VIEW). The script is executed on an empty SQLite database that has PRAGMA foreign_keys = ON. Use exactly the table, column and view names given here.

TABLES
customers(id INTEGER PRIMARY KEY, email, name)
  - email and name are required text; email must contain '@'; two customers may not have emails that differ only in letter case.
products(id INTEGER PRIMARY KEY, sku, price_cents, stock)
  - sku is required and unique; price_cents is a required integer >= 0; stock is a required integer >= 0 with default 0.
orders(id INTEGER PRIMARY KEY, customer_id, status)
  - customer_id is required and must reference an existing customer; a customer that has orders cannot be deleted.
  - status is required, one of 'open', 'paid', 'cancelled', with default 'open'.
order_items(order_id, product_id, qty, unit_price_cents)
  - order_id and product_id are required and must reference existing rows; at most one row per (order_id, product_id).
  - qty is a required integer > 0; unit_price_cents is a required integer >= 0.

BEHAVIOUR (use triggers and/or constraints; a statement that breaks a rule must fail with an error and change nothing)
R1. Inserting an order item decreases the product's stock by qty; it must fail when the stock is insufficient.
R2. Deleting an order item gives its qty back to the product's stock.
R3. Changing the qty of an order item adjusts the stock by the difference; it must fail when the stock is insufficient.
R4. Order items may only be inserted, updated or deleted while their order is 'open'.
R5. Allowed status changes: open -> paid, open -> cancelled, paid -> cancelled. Every other change of status must fail.
R6. When an order becomes 'cancelled', the stock of all its items is given back (the item rows stay).

VIEWS
order_totals(order_id, customer_id, status, item_count, total_cents): one row per order, including orders without items. item_count = number of item rows; total_cents = sum of qty * unit_price_cents; both are 0 when the order has no items.
customer_summary(customer_id, email, open_orders, paid_total_cents): one row per customer, including customers without orders. open_orders = number of that customer's orders whose status is 'open'; paid_total_cents = sum of total_cents over that customer's 'paid' orders (0 when there are none).

INDEX
Create an index that makes looking up orders by customer_id efficient."""

SHOP_SOLUTION = r"""
CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  email TEXT NOT NULL COLLATE NOCASE UNIQUE CHECK (instr(email, '@') > 0),
  name TEXT NOT NULL
);
CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  sku TEXT NOT NULL UNIQUE,
  price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
  stock INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0)
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'paid', 'cancelled'))
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE TABLE order_items (
  order_id INTEGER NOT NULL REFERENCES orders(id),
  product_id INTEGER NOT NULL REFERENCES products(id),
  qty INTEGER NOT NULL CHECK (qty > 0),
  unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0),
  PRIMARY KEY (order_id, product_id)
);
CREATE TRIGGER oi_bi BEFORE INSERT ON order_items
BEGIN
  SELECT RAISE(ABORT, 'order is not open')
   WHERE (SELECT status FROM orders WHERE id = NEW.order_id) IS NOT 'open';
  SELECT RAISE(ABORT, 'insufficient stock')
   WHERE (SELECT stock FROM products WHERE id = NEW.product_id) < NEW.qty;
END;
CREATE TRIGGER oi_ai AFTER INSERT ON order_items
BEGIN
  UPDATE products SET stock = stock - NEW.qty WHERE id = NEW.product_id;
END;
CREATE TRIGGER oi_bd BEFORE DELETE ON order_items
BEGIN
  SELECT RAISE(ABORT, 'order is not open')
   WHERE (SELECT status FROM orders WHERE id = OLD.order_id) IS NOT 'open';
END;
CREATE TRIGGER oi_ad AFTER DELETE ON order_items
BEGIN
  UPDATE products SET stock = stock + OLD.qty WHERE id = OLD.product_id;
END;
CREATE TRIGGER oi_bu BEFORE UPDATE ON order_items
BEGIN
  SELECT RAISE(ABORT, 'order is not open')
   WHERE (SELECT status FROM orders WHERE id = OLD.order_id) IS NOT 'open';
  SELECT RAISE(ABORT, 'insufficient stock')
   WHERE (SELECT stock FROM products WHERE id = NEW.product_id) < NEW.qty - OLD.qty;
END;
CREATE TRIGGER oi_au AFTER UPDATE OF qty ON order_items
BEGIN
  UPDATE products SET stock = stock - (NEW.qty - OLD.qty) WHERE id = NEW.product_id;
END;
CREATE TRIGGER o_bu BEFORE UPDATE OF status ON orders
WHEN NEW.status IS NOT OLD.status
 AND NOT ((OLD.status = 'open' AND NEW.status IN ('paid', 'cancelled'))
       OR (OLD.status = 'paid' AND NEW.status = 'cancelled'))
BEGIN
  SELECT RAISE(ABORT, 'invalid status change');
END;
CREATE TRIGGER o_au AFTER UPDATE OF status ON orders
WHEN NEW.status = 'cancelled' AND OLD.status <> 'cancelled'
BEGIN
  UPDATE products
     SET stock = stock + (SELECT qty FROM order_items WHERE order_id = NEW.id AND product_id = products.id)
   WHERE id IN (SELECT product_id FROM order_items WHERE order_id = NEW.id);
END;
CREATE VIEW order_totals AS
SELECT o.id AS order_id, o.customer_id, o.status,
       COUNT(i.product_id) AS item_count,
       COALESCE(SUM(i.qty * i.unit_price_cents), 0) AS total_cents
FROM orders o LEFT JOIN order_items i ON i.order_id = o.id
GROUP BY o.id;
CREATE VIEW customer_summary AS
SELECT c.id AS customer_id, c.email,
       COALESCE(SUM(t.status = 'open'), 0) AS open_orders,
       COALESCE(SUM(CASE WHEN t.status = 'paid' THEN t.total_cents END), 0) AS paid_total_cents
FROM customers c LEFT JOIN order_totals t ON t.customer_id = c.id
GROUP BY c.id;
"""

_SHOP_BASE = [
    ("do", "INSERT INTO customers(id, email, name) VALUES (1, 'ann@example.com', 'Ann')"),
    ("do", "INSERT INTO customers(id, email, name) VALUES (2, 'bob@example.com', 'Bob')"),
    ("do", "INSERT INTO products(id, sku, price_cents, stock) VALUES (1, 'SKU-1', 250, 10)"),
    ("do", "INSERT INTO products(id, sku, price_cents, stock) VALUES (2, 'SKU-2', 999, 5)"),
    ("do", "INSERT INTO orders(id, customer_id) VALUES (1, 1)"),
]
_STOCK = "SELECT id, stock FROM products ORDER BY id"
_ITEMS = "SELECT order_id, product_id, qty FROM order_items ORDER BY order_id, product_id"


def _shop_index_check(answer):
    schema = sql_statements(answer)
    conn = sql_connect()
    try:
        run_script(conn, schema)
        for row in conn.execute("PRAGMA index_list('orders')").fetchall():
            cols = conn.execute("PRAGMA index_info('%s')" % row[1].replace("'", "''")).fetchall()
            if cols and cols[0][2] == "customer_id":
                return True, ""
        return False, "no index on orders whose first column is customer_id"
    except Exception as e:
        return False, "script failed: %s" % shorten(str(e), 200)
    finally:
        conn.close()


def shop_grade(answer, scale=1.0):
    B = _SHOP_BASE
    checks = [
        ("customers: required columns and '@' check", B + [
            ("fail", "INSERT INTO customers(id, email, name) VALUES (3, NULL, 'X')"),
            ("fail", "INSERT INTO customers(id, email, name) VALUES (3, 'x@example.com', NULL)"),
            ("fail", "INSERT INTO customers(id, email, name) VALUES (3, 'no-at-sign', 'X')"),
            ("fail", "INSERT INTO customers(id, email, name) VALUES (3, 'bob@example.com', 'Other Bob')"),
            ("do", "INSERT INTO customers(id, email, name) VALUES (3, 'cy@example.com', 'Cy')")]),
        ("customers: e-mail unique regardless of letter case", B + [
            ("fail", "INSERT INTO customers(id, email, name) VALUES (3, 'ANN@Example.COM', 'Ann 2')"),
            ("do", "INSERT INTO customers(id, email, name) VALUES (4, 'ann2@example.com', 'Ann 3')")]),
        ("products: constraints and default stock", B + [
            ("fail", "INSERT INTO products(id, sku, price_cents, stock) VALUES (3, 'SKU-1', 5, 1)"),
            ("fail", "INSERT INTO products(id, sku, price_cents, stock) VALUES (3, 'SKU-3', -1, 1)"),
            ("fail", "INSERT INTO products(id, sku, price_cents, stock) VALUES (3, 'SKU-3', 5, -1)"),
            ("fail", "INSERT INTO products(id, sku, price_cents, stock) VALUES (3, NULL, 5, 1)"),
            ("fail", "UPDATE products SET stock = -3 WHERE id = 1"),
            ("do", "INSERT INTO products(id, sku, price_cents) VALUES (3, 'SKU-3', 0)"),
            ("rows", "SELECT stock FROM products WHERE id = 3", [(0,)])]),
        ("orders: status values, default and foreign key", B + [
            ("fail", "INSERT INTO orders(id, customer_id, status) VALUES (2, 1, 'shipped')"),
            ("fail", "INSERT INTO orders(id, customer_id) VALUES (2, 99)"),
            ("fail", "INSERT INTO orders(id, customer_id) VALUES (2, NULL)"),
            ("fail", "DELETE FROM customers WHERE id = 1"),
            ("do", "DELETE FROM customers WHERE id = 2"),
            ("rows", "SELECT status FROM orders WHERE id = 1", [("open",)])]),
        ("order_items: constraints and foreign keys", B + [
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 0, 250)"),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, -2, 250)"),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 1, -5)"),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 77, 1, 250)"),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (55, 1, 1, 250)"),
            ("rows", _STOCK, [(1, 10), (2, 5)]),
            ("rows", "SELECT COUNT(*) FROM order_items", [(0,)])]),
        ("R1: insert decreases stock, insufficient stock fails", B + [
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("rows", _STOCK, [(1, 6), (2, 5)]),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 6, 999)"),
            ("rows", _STOCK, [(1, 6), (2, 5)]),
            ("rows", _ITEMS, [(1, 1, 4)]),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 5, 999)"),
            ("rows", _STOCK, [(1, 6), (2, 0)])]),
        ("one row per (order, product), stock untouched by the rejected insert", B + [
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 2, 250)"),
            ("rows", _STOCK, [(1, 6), (2, 5)]),
            ("rows", _ITEMS, [(1, 1, 4)])]),
        ("R2: delete gives stock back", B + [
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 2, 999)"),
            ("do", "DELETE FROM order_items WHERE order_id = 1 AND product_id = 1"),
            ("rows", _STOCK, [(1, 10), (2, 3)])]),
        ("R3: changing qty adjusts stock", B + [
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("do", "UPDATE order_items SET qty = 7 WHERE order_id = 1 AND product_id = 1"),
            ("rows", _STOCK, [(1, 3), (2, 5)]),
            ("fail", "UPDATE order_items SET qty = 20 WHERE order_id = 1 AND product_id = 1"),
            ("rows", _STOCK, [(1, 3), (2, 5)]),
            ("rows", _ITEMS, [(1, 1, 7)]),
            ("do", "UPDATE order_items SET qty = 2 WHERE order_id = 1 AND product_id = 1"),
            ("rows", _STOCK, [(1, 8), (2, 5)]),
            ("do", "UPDATE order_items SET qty = 10 WHERE order_id = 1 AND product_id = 1"),
            ("rows", _STOCK, [(1, 0), (2, 5)])]),
        ("R4: items are frozen once the order is not open", B + [
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("do", "UPDATE orders SET status = 'paid' WHERE id = 1"),
            ("fail", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 1, 999)"),
            ("fail", "UPDATE order_items SET qty = 5 WHERE order_id = 1 AND product_id = 1"),
            ("fail", "DELETE FROM order_items WHERE order_id = 1 AND product_id = 1"),
            ("rows", _STOCK, [(1, 6), (2, 5)]),
            ("rows", _ITEMS, [(1, 1, 4)])]),
        ("R5: allowed status changes", B + [
            ("do", "INSERT INTO orders(id, customer_id) VALUES (2, 2)"),
            ("do", "INSERT INTO orders(id, customer_id) VALUES (3, 2)"),
            ("do", "UPDATE orders SET status = 'paid' WHERE id = 1"),
            ("do", "UPDATE orders SET status = 'cancelled' WHERE id = 1"),
            ("do", "UPDATE orders SET status = 'cancelled' WHERE id = 2"),
            ("rows", "SELECT id, status FROM orders ORDER BY id", [(1, "cancelled"), (2, "cancelled"), (3, "open")])]),
        ("R5: forbidden status changes", B + [
            ("do", "INSERT INTO orders(id, customer_id) VALUES (2, 2)"),
            ("do", "UPDATE orders SET status = 'paid' WHERE id = 1"),
            ("fail", "UPDATE orders SET status = 'open' WHERE id = 1"),
            ("do", "UPDATE orders SET status = 'cancelled' WHERE id = 2"),
            ("fail", "UPDATE orders SET status = 'paid' WHERE id = 2"),
            ("fail", "UPDATE orders SET status = 'open' WHERE id = 2"),
            ("rows", "SELECT id, status FROM orders ORDER BY id", [(1, "paid"), (2, "cancelled")])]),
        ("R6: cancelling gives stock back, items stay", B + [
            ("do", "INSERT INTO orders(id, customer_id) VALUES (2, 2)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 2, 999)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (2, 1, 3, 250)"),
            ("do", "UPDATE orders SET status = 'cancelled' WHERE id = 1"),
            ("rows", _STOCK, [(1, 7), (2, 5)]),
            ("rows", _ITEMS, [(1, 1, 4), (1, 2, 2), (2, 1, 3)]),
            ("do", "UPDATE orders SET status = 'paid' WHERE id = 2"),
            ("rows", _STOCK, [(1, 7), (2, 5)]),
            ("do", "UPDATE orders SET status = 'cancelled' WHERE id = 2"),
            ("rows", _STOCK, [(1, 10), (2, 5)])]),
        ("view order_totals", B + [
            ("do", "INSERT INTO orders(id, customer_id) VALUES (2, 2)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 2, 999)"),
            ("rows", "SELECT order_id, customer_id, status, item_count, total_cents FROM order_totals ORDER BY order_id",
             [(1, 1, "open", 2, 2998), (2, 2, "open", 0, 0)])]),
        ("view customer_summary", B + [
            ("do", "INSERT INTO customers(id, email, name) VALUES (3, 'cy@example.com', 'Cy')"),
            ("do", "INSERT INTO orders(id, customer_id) VALUES (2, 1)"),
            ("do", "INSERT INTO orders(id, customer_id) VALUES (3, 1)"),
            ("do", "INSERT INTO orders(id, customer_id) VALUES (4, 1)"),
            ("do", "INSERT INTO orders(id, customer_id) VALUES (5, 3)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 1, 4, 250)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (1, 2, 2, 999)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (2, 1, 1, 100)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (3, 1, 2, 250)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (3, 2, 1, 999)"),
            ("do", "INSERT INTO order_items(order_id, product_id, qty, unit_price_cents) VALUES (5, 2, 1, 999)"),
            ("do", "UPDATE orders SET status = 'paid' WHERE id IN (1, 2)"),
            ("do", "UPDATE orders SET status = 'cancelled' WHERE id = 5"),
            ("rows", "SELECT customer_id, email, open_orders, paid_total_cents FROM customer_summary ORDER BY customer_id",
             [(1, "ann@example.com", 2, 3098), (2, "bob@example.com", 0, 0), (3, "cy@example.com", 0, 0)])]),
    ]
    results = grade_behaviour(answer, checks)
    results.append(R("index on orders(customer_id)", *_shop_index_check(answer), weight=0.5))
    return results


task(id="sql_schema_shop", lang="sql", category="sql",
     title="Design a schema with constraints, triggers and views",
     spec=SHOP_SPEC, grade=shop_grade, solution=SHOP_SOLUTION, raw_prompt=True)


# ---- SQL 2: recursive query (bill of materials) ----------------------------

BOM_SCHEMA = [
    "CREATE TABLE parts (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, unit_cost_cents INTEGER NOT NULL)",
    "CREATE TABLE bom (parent_id INTEGER NOT NULL REFERENCES parts(id), child_id INTEGER NOT NULL REFERENCES parts(id), qty INTEGER NOT NULL)",
]

BOM_SPEC = r"""A SQLite database has these tables:

  CREATE TABLE parts (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, unit_cost_cents INTEGER NOT NULL);
  CREATE TABLE bom (parent_id INTEGER NOT NULL REFERENCES parts(id),
                    child_id  INTEGER NOT NULL REFERENCES parts(id),
                    qty       INTEGER NOT NULL);

A bom row means "one unit of parent_id needs qty units of child_id". The same (parent_id, child_id) pair may appear in more than one row; their quantities add up. The graph has no cycles.
A ROOT is a part that never appears as a child. A LEAF is a part that never appears as a parent. A part can be both.

Write ONE SELECT statement (it may start with WITH) that returns, for every root, every leaf needed to build one unit of that root:
  root_name, leaf_name, total_qty, total_cost_cents
total_qty is the total number of units of the leaf needed for one unit of the root: multiply the quantities along each path from the root to the leaf and add up all paths (a leaf can be reached through several different paths, and different paths can have exactly the same quantities). total_cost_cents = total_qty * the leaf's unit_cost_cents. A root that is also a leaf yields a single row with itself as the leaf and total_qty 1.
Order the result by root_name, then leaf_name.

Answer with the SELECT statement only, in one ```sql block."""

BOM_SOLUTION = r"""
WITH RECURSIVE walk(root_id, part_id, qty) AS (
  SELECT p.id, p.id, 1 FROM parts p WHERE p.id NOT IN (SELECT child_id FROM bom)
  UNION ALL
  SELECT w.root_id, b.child_id, w.qty * b.qty FROM walk w JOIN bom b ON b.parent_id = w.part_id
)
SELECT r.name AS root_name, l.name AS leaf_name, SUM(w.qty) AS total_qty,
       SUM(w.qty) * l.unit_cost_cents AS total_cost_cents
FROM walk w JOIN parts r ON r.id = w.root_id JOIN parts l ON l.id = w.part_id
WHERE w.part_id NOT IN (SELECT parent_id FROM bom)
GROUP BY r.id, l.id
ORDER BY r.name, l.name;
"""


def _bom_expected(parts, bom):
    children = {}
    for p, c, q in bom:
        children.setdefault(p, []).append((c, q))
    child_ids = set(c for _, c, _ in bom)
    name = {pid: nm for pid, nm, _ in parts}
    cost = {pid: cst for pid, _, cst in parts}
    memo = {}

    def need(pid):
        if pid in memo:
            return memo[pid]
        if pid not in children:
            res = {pid: 1}
        else:
            res = {}
            for c, q in children[pid]:
                for leaf, n in need(c).items():
                    res[leaf] = res.get(leaf, 0) + n * q
        memo[pid] = res
        return res

    rows = []
    for pid, nm, _ in parts:
        if pid in child_ids:
            continue
        for leaf, n in need(pid).items():
            rows.append((nm, name[leaf], n, n * cost[leaf]))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def _bom_dataset(name, parts, bom):
    setup = list(BOM_SCHEMA)
    setup.append(("INSERT INTO parts VALUES (?, ?, ?)", parts))
    setup.append(("INSERT INTO bom VALUES (?, ?, ?)", bom))
    return (name, setup, _bom_expected(parts, bom))


def _bom_random(seed, layers, width):
    r = Rng(seed)
    parts, bom, by_layer = [], [], []
    pid = 0
    for layer in range(layers):
        ids = []
        for _ in range(r.randint(2, width)):
            pid += 1
            parts.append((pid, "%s-%03d" % (r.choice(["bolt", "frame", "panel", "unit", "kit"]), (pid * 37) % 1000),
                          r.randint(1, 500)))
            ids.append(pid)
        by_layer.append(ids)
    for layer in range(layers - 1):
        lower = [x for ids in by_layer[layer + 1:] for x in ids]
        for p in by_layer[layer]:
            if r.chance(12):
                continue                       # stays a leaf (and maybe a root)
            for _ in range(r.randint(1, 3)):
                bom.append((p, r.choice(lower), r.randint(1, 4)))
    return parts, bom


def bom_grade(answer, scale=1.0):
    chain_parts = [(i, "c%02d" % i, 3) for i in range(1, 41)]
    chain_bom = [(i, i + 1, 2 if i % 2 else 1) for i in range(1, 40)]
    datasets = [
        _bom_dataset("simple tree",
                     [(1, "bike", 0), (2, "wheel", 0), (3, "frame", 9000), (4, "spoke", 20), (5, "rim", 1500)],
                     [(1, 2, 2), (1, 3, 1), (2, 4, 32), (2, 5, 1)]),
        _bom_dataset("leaf reached through several paths",
                     [(1, "cart", 0), (2, "axle", 0), (3, "body", 0), (4, "bolt", 5), (5, "wheel", 700)],
                     [(1, 2, 2), (1, 3, 1), (2, 4, 4), (3, 4, 10), (2, 5, 2), (1, 4, 3)]),
        _bom_dataset("parallel paths with identical quantities",
                     [(1, "desk", 0), (2, "left-leg", 0), (3, "right-leg", 0), (4, "screw", 2), (5, "top", 4000)],
                     [(1, 2, 1), (1, 3, 1), (2, 4, 4), (3, 4, 4), (1, 5, 1), (1, 4, 4), (1, 4, 4)]),
        _bom_dataset("deep chain", chain_parts, chain_bom),
        _bom_dataset("several roots, shared sub-assemblies, stand-alone part",
                     [(1, "kit-a", 0), (2, "kit-b", 0), (3, "module", 0), (4, "chip", 250), (5, "case", 90),
                      (6, "manual", 15), (7, "cable", 40)],
                     [(1, 3, 2), (2, 3, 5), (3, 4, 3), (3, 5, 1), (1, 7, 1), (2, 4, 1)]),
        _bom_dataset("random assembly 1", *_bom_random(2001, 4, 4)),
        _bom_dataset("random assembly 2", *_bom_random(2002, 5, 5)),
        _bom_dataset("random assembly 3", *_bom_random(2003, 6, 6)),
    ]
    return grade_query(answer, datasets)


task(id="sql_query_bom", lang="sql", category="sql",
     title="Recursive query: bill of materials explosion",
     spec=BOM_SPEC, grade=bom_grade, solution=BOM_SOLUTION, raw_prompt=True)


# ---- SQL 3: window functions (login streaks) -------------------------------

STREAK_SCHEMA = ["CREATE TABLE logins (user_id INTEGER NOT NULL, day INTEGER NOT NULL)"]

STREAK_SPEC = r"""A SQLite database has this table:

  CREATE TABLE logins (user_id INTEGER NOT NULL, day INTEGER NOT NULL);

Each row records one login. day is a day number (an integer; consecutive calendar days have consecutive numbers). A user can log in many times on the same day.

Write ONE SELECT statement (it may start with WITH) that returns, for every user present in the table, the longest streak of consecutive days on which the user logged in at least once:
  user_id, streak_start, streak_end, streak_days
streak_start and streak_end are the first and the last day number of the streak; streak_days is the number of days in it. When a user has several longest streaks of the same length, report the most recent one.
Order the result by streak_days descending, then user_id ascending.

Answer with the SELECT statement only, in one ```sql block."""

STREAK_SOLUTION = r"""
WITH d AS (SELECT DISTINCT user_id, day FROM logins),
g AS (SELECT user_id, day, day - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY day) AS grp FROM d),
s AS (SELECT user_id, MIN(day) AS s, MAX(day) AS e, COUNT(*) AS n FROM g GROUP BY user_id, grp),
r AS (SELECT s.*, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY n DESC, s DESC) AS rn FROM s)
SELECT user_id, s AS streak_start, e AS streak_end, n AS streak_days
FROM r WHERE rn = 1
ORDER BY n DESC, user_id;
"""


def _streak_expected(rows):
    days = {}
    for uid, day in rows:
        days.setdefault(uid, set()).add(day)
    out = []
    for uid, ds in days.items():
        ordered = sorted(ds)
        best = None
        start = prev = ordered[0]
        for d in ordered[1:] + [None]:
            if d is not None and d - prev == 1:
                prev = d
                continue
            length = prev - start + 1
            if best is None or length >= best[2]:
                best = (start, prev, length)
            if d is not None:
                start = prev = d
        out.append((uid, best[0], best[1], best[2]))
    out.sort(key=lambda r: (-r[3], r[0]))
    return out


def _streak_dataset(name, rows):
    return (name, STREAK_SCHEMA + [("INSERT INTO logins VALUES (?, ?)", rows)], _streak_expected(rows))


def _streak_random(seed, users, span, repeats=3):
    r = Rng(seed)
    rows = []
    for uid in range(1, users + 1):
        day = 700 + r.randint(0, 5)
        for _ in range(r.randint(1, span)):
            day += 1 if r.chance(72) else r.randint(2, 6)
            for _ in range(r.randint(1, repeats)):
                rows.append((uid * 7 % 101, day))
    r.shuffle(rows)
    return rows


def streak_grade(answer, scale=1.0):
    # Streak lengths differ between users unless a test is about ties, and only the later
    # tests repeat a day: a query without DISTINCT still earns the first ones.
    datasets = [
        _streak_dataset("one streak per user", [(1, 10), (1, 11), (1, 12), (2, 20), (2, 21), (3, 5)]),
        _streak_dataset("several streaks per user", [(1, 10), (1, 11), (1, 12), (1, 16), (2, 20),
                                                     (2, 24), (2, 25), (2, 26), (2, 27), (2, 30),
                                                     (2, 31), (3, 7), (3, 9)]),
        _streak_dataset("rows in any order", [(9, 103), (4, 101), (7, 102), (9, 101), (9, 102),
                                              (9, 104), (12, 109), (4, 102), (9, 99), (7, 101),
                                              (7, 100), (7, 103), (7, 104)]),
        _streak_dataset("equal streaks: most recent wins", [(1, 1), (1, 2), (1, 5), (1, 6), (1, 9),
                                                            (2, 10), (2, 12), (2, 14), (3, 1),
                                                            (3, 2), (3, 3)]),
        _streak_dataset("ordered by length, then user", [(5, 1), (5, 2), (3, 4), (3, 5), (8, 7),
                                                         (2, 1), (2, 2), (2, 3), (1, 9), (4, 30),
                                                         (4, 31)]),
        _streak_dataset("several logins per day", [(1, 40), (1, 40), (1, 41), (1, 41), (1, 41),
                                                   (2, 40), (2, 40), (2, 40), (1, 43), (3, 50),
                                                   (3, 51), (3, 51), (3, 52)]),
        _streak_dataset("random, one login per day", _streak_random(2101, 30, 40, repeats=1)),
        _streak_dataset("random 1", _streak_random(2102, 40, 60)),
        _streak_dataset("random 2", _streak_random(2103, 90, 120)),
    ]
    return grade_query(answer, datasets)


task(id="sql_query_streaks", lang="sql", category="sql",
     title="Window functions: longest login streak per user",
     spec=STREAK_SPEC, grade=streak_grade, solution=STREAK_SOLUTION, raw_prompt=True)


# ---- SQL 4: a report query with hidden bugs --------------------------------

REPORT_SCHEMA = [
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)",
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id), status TEXT, created_at TEXT NOT NULL)",
    "CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id), product TEXT NOT NULL, qty INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL)",
    "CREATE TABLE payments (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id), amount_cents INTEGER NOT NULL)",
]

REPORT_GIVEN = r"""SELECT c.name,
       COUNT(o.id)                     AS orders,
       SUM(i.qty * i.unit_price_cents) AS items_cents,
       SUM(p.amount_cents)             AS paid_cents
FROM customers c
LEFT JOIN orders o      ON o.customer_id = c.id
LEFT JOIN order_items i ON i.order_id = o.id
LEFT JOIN payments p    ON p.order_id = o.id
WHERE o.status <> 'cancelled'
  AND o.created_at BETWEEN '2026-03-01' AND '2026-03-31'
GROUP BY c.id
ORDER BY items_cents DESC, c.name;
"""

REPORT_SPEC = r"""A SQLite database has these tables:

  CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
  CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                       status TEXT, created_at TEXT NOT NULL);       -- created_at: 'YYYY-MM-DD HH:MM:SS'
  CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id), product TEXT NOT NULL,
                            qty INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL);
  CREATE TABLE payments (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id),
                         amount_cents INTEGER NOT NULL);

REQUIRED REPORT (March 2026)
One row per customer - ALL customers, including those without any qualifying order - with the columns:
  name, orders, items_cents, paid_cents
A qualifying order is an order of that customer created from 2026-03-01 00:00:00 through 2026-03-31 23:59:59 (inclusive) whose status is not 'cancelled'. status can be NULL (old rows); orders with NULL status DO qualify.
orders = number of qualifying orders; items_cents = sum of qty * unit_price_cents over all items of the qualifying orders; paid_cents = sum of amount_cents over all payments of the qualifying orders. An order can have any number of items and any number of payments, including none. Totals are 0 (never NULL) when there is nothing to add.
Sort by items_cents descending, then name ascending.

The query below was written for this report. It runs, and on simple data it even looks right, but it contains several bugs (how many is not disclosed). Find and fix ALL of them.

```sql
""" + REPORT_GIVEN + r"""```

Answer with the corrected SELECT statement only, in one ```sql block."""

REPORT_SOLUTION = r"""
SELECT c.name,
       COUNT(o.id) AS orders,
       COALESCE(SUM((SELECT SUM(i.qty * i.unit_price_cents) FROM order_items i WHERE i.order_id = o.id)), 0) AS items_cents,
       COALESCE(SUM((SELECT SUM(p.amount_cents) FROM payments p WHERE p.order_id = o.id)), 0) AS paid_cents
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
                  AND (o.status IS NULL OR o.status <> 'cancelled')
                  AND o.created_at >= '2026-03-01' AND o.created_at < '2026-04-01'
GROUP BY c.id
ORDER BY items_cents DESC, c.name;
"""


def _report_expected(customers, orders, items, payments):
    item_sum, pay_sum = {}, {}
    for oid, _, qty, price in items:
        item_sum[oid] = item_sum.get(oid, 0) + qty * price
    for _, oid, amt in payments:
        pay_sum[oid] = pay_sum.get(oid, 0) + amt
    agg = {cid: [0, 0, 0] for cid, _ in customers}
    for oid, cid, status, created in orders:
        if status == "cancelled":
            continue
        if not ("2026-03-01 00:00:00" <= created <= "2026-03-31 23:59:59"):
            continue
        a = agg[cid]
        a[0] += 1
        a[1] += item_sum.get(oid, 0)
        a[2] += pay_sum.get(oid, 0)
    rows = [(name, agg[cid][0], agg[cid][1], agg[cid][2]) for cid, name in customers]
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows


def _report_dataset(name, customers, orders, items, payments):
    setup = list(REPORT_SCHEMA)
    setup.append(("INSERT INTO customers VALUES (?, ?)", customers))
    setup.append(("INSERT INTO orders VALUES (?, ?, ?, ?)", orders))
    setup.append(("INSERT INTO order_items VALUES (?, ?, ?, ?)", items))
    setup.append(("INSERT INTO payments VALUES (?, ?, ?)", payments))
    return (name, setup, _report_expected(customers, orders, items, payments))


def _report_random(seed, ncust, norders):
    r = Rng(seed)
    customers = [(i, "%s %c." % (r.choice(["Ana", "Bruno", "Carla", "Davi", "Elisa", "Fabio"]), 64 + i))
                 for i in range(1, ncust + 1)]
    orders, items, payments = [], [], []
    pid = 0
    for oid in range(1, norders + 1):
        day = r.choice(["2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02", "2026-03-15", "2026-03-30",
                        "2026-03-31", "2026-03-31", "2026-04-01", "2026-04-02", "2025-03-15", "2026-03-09"])
        t = r.choice(["00:00:00", "23:59:59", "12:00:00", "08:15:30"])
        orders.append((oid, r.randint(1, max(1, ncust - 1)), r.choice(["paid", "open", "cancelled", None, "paid"]),
                       "%s %s" % (day, t)))
        for k in range(r.randint(0, 3)):
            items.append((oid, "p%d" % k, r.randint(1, 5), r.randint(100, 9000)))
        for _ in range(r.randint(0, 3)):
            pid += 1
            payments.append((pid, oid, r.randint(100, 20000)))
    return customers, orders, items, payments


def report_grade(answer, scale=1.0):
    C = [(1, "Ana"), (2, "Bruno"), (3, "Carla")]
    datasets = [
        _report_dataset("simple data", C,
                        [(1, 1, "paid", "2026-03-05 10:00:00"), (2, 2, "open", "2026-03-10 09:30:00"),
                         (3, 3, "paid", "2026-03-20 18:00:00"), (4, 1, "cancelled", "2026-03-06 10:00:00"),
                         (5, 2, "paid", "2026-02-10 10:00:00"), (6, 3, "paid", "2026-03-21 08:00:00")],
                        [(1, "pen", 2, 500), (2, "ink", 1, 1200), (3, "pad", 3, 700), (4, "pen", 9, 500),
                         (5, "pen", 1, 500), (6, "cap", 1, 300)],
                        [(1, 1, 1000), (2, 2, 1200), (3, 3, 2100), (4, 4, 4500), (5, 5, 500), (6, 6, 300)]),
        _report_dataset("customers without qualifying orders", C + [(4, "Davi"), (5, "Elisa")],
                        [(1, 1, "paid", "2026-03-05 10:00:00"), (2, 2, "cancelled", "2026-03-10 09:30:00"),
                         (3, 3, "paid", "2026-01-20 18:00:00"), (4, 5, "paid", "2026-03-11 12:00:00")],
                        [(1, "pen", 2, 500), (2, "ink", 1, 1200), (3, "pad", 3, 700), (4, "ink", 1, 900)],
                        [(1, 1, 1000), (2, 2, 1200), (3, 3, 2100), (4, 4, 900)]),
        _report_dataset("orders with NULL status", C,
                        [(1, 1, None, "2026-03-05 10:00:00"), (2, 2, "open", "2026-03-10 09:30:00"),
                         (3, 3, None, "2026-03-20 18:00:00"), (4, 3, "paid", "2026-03-21 18:00:00")],
                        [(1, "pen", 2, 500), (2, "ink", 1, 1200), (3, "pad", 3, 700), (4, "pad", 1, 700)],
                        [(1, 1, 1000), (2, 2, 1200), (3, 3, 2100), (4, 4, 700)]),
        _report_dataset("last day of the month and neighbours", C,
                        [(1, 1, "paid", "2026-03-31 10:00:00"), (2, 2, "open", "2026-03-31 23:59:59"),
                         (3, 3, "paid", "2026-04-01 00:00:00"), (4, 1, "paid", "2026-03-01 00:00:00"),
                         (5, 2, "paid", "2026-02-28 23:59:59"), (6, 3, "paid", "2026-03-31 00:00:00"),
                         (7, 1, "paid", "2026-03-10 00:00:00"), (8, 2, "paid", "2026-03-10 00:00:00")],
                        [(i, "pen", 1, 100 * i) for i in range(1, 9)],
                        [(i, i, 100 * i) for i in range(1, 9)]),
        _report_dataset("order with several items", C,
                        [(1, 1, "paid", "2026-03-05 10:00:00"), (2, 2, "open", "2026-03-10 09:30:00"),
                         (3, 3, "paid", "2026-03-20 18:00:00")],
                        [(1, "pen", 2, 500), (1, "ink", 1, 1200), (1, "pad", 3, 700), (2, "ink", 1, 1200),
                         (3, "pad", 3, 700)],
                        [(1, 1, 4300), (2, 2, 1200), (3, 3, 2100)]),
        _report_dataset("order with several payments", C,
                        [(1, 1, "paid", "2026-03-05 10:00:00"), (2, 2, "open", "2026-03-10 09:30:00"),
                         (3, 3, "paid", "2026-03-20 18:00:00")],
                        [(1, "pen", 2, 500), (2, "ink", 1, 1200), (3, "pad", 3, 700)],
                        [(1, 1, 400), (2, 1, 600), (3, 2, 1200), (4, 3, 100), (5, 3, 1000), (6, 3, 1000)]),
        _report_dataset("several items and several payments", C,
                        [(1, 1, "paid", "2026-03-05 10:00:00"), (2, 1, "paid", "2026-03-06 10:00:00"),
                         (3, 2, "open", "2026-03-10 09:30:00"), (4, 3, "paid", "2026-03-20 18:00:00")],
                        [(1, "pen", 2, 500), (1, "ink", 1, 1200), (2, "pad", 1, 700), (2, "cap", 2, 300),
                         (2, "pen", 1, 500), (3, "ink", 1, 1200), (4, "pad", 3, 700)],
                        [(1, 1, 1000), (2, 1, 1200), (3, 2, 900), (4, 2, 900), (5, 3, 1200), (6, 4, 2100)]),
        _report_dataset("orders without items or without payments", C,
                        [(1, 1, "paid", "2026-03-05 10:00:00"), (2, 2, "open", "2026-03-10 09:30:00"),
                         (3, 3, "paid", "2026-03-20 18:00:00"), (4, 3, "open", "2026-03-22 18:00:00")],
                        [(1, "pen", 2, 500), (4, "pad", 1, 50)],
                        [(1, 2, 1200)]),
        _report_dataset("random 1", *_report_random(2201, 6, 25)),
        _report_dataset("random 2", *_report_random(2202, 9, 80)),
        _report_dataset("random 3", *_report_random(2203, 14, 300)),
    ]
    return grade_query(answer, datasets)


task(id="sql_bugfix_report", lang="sql", category="sql",
     title="Report query with hidden bugs",
     spec=REPORT_SPEC, grade=report_grade, solution=REPORT_SOLUTION, given=REPORT_GIVEN, raw_prompt=True)


# ---- SQL 5: a schema/trigger script with hidden bugs -----------------------

LEDGER_GIVEN = r"""CREATE TABLE accounts (
    id            INTEGER PRIMARY KEY,
    owner         TEXT NOT NULL UNIQUE,
    currency      TEXT NOT NULL CHECK (length(currency) <= 3),
    balance_cents INTEGER NOT NULL DEFAULT 0 CHECK (balance_cents >= 0),
    closed        INTEGER NOT NULL DEFAULT 0 CHECK (closed IN (0, 1))
);

CREATE TABLE transfers (
    id           INTEGER PRIMARY KEY,
    from_id      INTEGER NOT NULL REFERENCES accounts(id),
    to_id        INTEGER NOT NULL REFERENCES accounts(id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL,
    old_balance INTEGER NOT NULL,
    new_balance INTEGER NOT NULL
);

CREATE TRIGGER transfers_check BEFORE INSERT ON transfers
BEGIN
    SELECT RAISE(ABORT, 'account is closed')
     WHERE (SELECT closed FROM accounts WHERE id = NEW.from_id) = 1;
    SELECT RAISE(ABORT, 'currency mismatch')
     WHERE (SELECT currency FROM accounts WHERE id = NEW.from_id)
        <> (SELECT currency FROM accounts WHERE id = NEW.from_id);
END;

CREATE TRIGGER transfers_apply AFTER INSERT ON transfers
BEGIN
    UPDATE accounts SET balance_cents = balance_cents - NEW.amount_cents WHERE id = NEW.from_id;
    UPDATE accounts SET balance_cents = balance_cents + NEW.amount_cents WHERE id = NEW.to_id;
END;

CREATE TRIGGER transfers_no_delete BEFORE DELETE ON transfers
BEGIN
    SELECT RAISE(ABORT, 'transfers are immutable');
END;

CREATE TRIGGER accounts_audit AFTER UPDATE ON accounts
BEGIN
    INSERT INTO audit_log (account_id, old_balance, new_balance)
    VALUES (NEW.id, OLD.balance_cents, NEW.balance_cents);
END;

CREATE TRIGGER accounts_close BEFORE UPDATE OF closed ON accounts
WHEN NEW.closed = 1 AND OLD.balance_cents > 0
BEGIN
    SELECT RAISE(ABORT, 'balance must be zero');
END;

CREATE VIEW account_statement AS
SELECT a.id AS account_id,
       a.owner,
       SUM(tin.amount_cents)  AS total_in,
       SUM(tout.amount_cents) AS total_out,
       COUNT(*)               AS transfer_count
FROM accounts a
LEFT JOIN transfers tin  ON tin.to_id = a.id
LEFT JOIN transfers tout ON tout.from_id = a.id
GROUP BY a.id;
"""

LEDGER_SPEC = r"""The SQLite script below implements a small money ledger. It is executed on an empty database with PRAGMA foreign_keys = ON. It runs without errors, but it does NOT fully implement the requirements: it contains several bugs and omissions (how many is not disclosed). Fix ALL of them. Keep every table, column and view name.

REQUIREMENTS
Q1. accounts(id, owner, currency, balance_cents, closed): owner and currency are required; currency has exactly 3 characters; one owner may have several accounts, but at most one per currency; balance_cents can never be negative; closed is 0 or 1 (default 0).
Q2. transfers(id, from_id, to_id, amount_cents, created_at): both accounts must exist and must be different accounts; amount_cents must be greater than 0.
Q3. Inserting a transfer moves amount_cents from the from-account to the to-account. The insert must fail, changing nothing, when the from-account's balance is insufficient, when the two accounts have different currencies, or when either account is closed.
Q4. Transfers are immutable: every UPDATE or DELETE on transfers must fail.
Q5. audit_log receives one row (account_id, old_balance, new_balance) every time an account's balance actually changes - and only then (renaming an owner or closing an account must not add rows).
Q6. An account can only be closed (closed going from 0 to 1) when its balance is 0, and a closed account can never be reopened.
Q7. View account_statement(account_id, owner, total_in, total_out, transfer_count): one row per account; total_in = sum of the amounts received, total_out = sum of the amounts sent, transfer_count = number of transfers in which the account takes part. All three are 0 for an account without transfers.

```sql
""" + LEDGER_GIVEN + r"""```

Answer with the complete corrected script in one ```sql block."""

LEDGER_FIXES = [
    ("    owner         TEXT NOT NULL UNIQUE,", "    owner         TEXT NOT NULL,"),
    ("CHECK (length(currency) <= 3)", "CHECK (length(currency) = 3)"),
    ("    closed        INTEGER NOT NULL DEFAULT 0 CHECK (closed IN (0, 1))\n);",
     "    closed        INTEGER NOT NULL DEFAULT 0 CHECK (closed IN (0, 1)),\n    UNIQUE (owner, currency)\n);"),
    ("CHECK (amount_cents >= 0),\n    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n);",
     "CHECK (amount_cents > 0),\n    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n    CHECK (from_id <> to_id)\n);"),
    ("     WHERE (SELECT closed FROM accounts WHERE id = NEW.from_id) = 1;",
     "     WHERE (SELECT closed FROM accounts WHERE id = NEW.from_id) = 1\n"
     "        OR (SELECT closed FROM accounts WHERE id = NEW.to_id) = 1;"),
    ("        <> (SELECT currency FROM accounts WHERE id = NEW.from_id);",
     "        <> (SELECT currency FROM accounts WHERE id = NEW.to_id);"),
    ("    SELECT RAISE(ABORT, 'transfers are immutable');\nEND;",
     "    SELECT RAISE(ABORT, 'transfers are immutable');\nEND;\n\n"
     "CREATE TRIGGER transfers_no_update BEFORE UPDATE ON transfers\nBEGIN\n"
     "    SELECT RAISE(ABORT, 'transfers are immutable');\nEND;"),
    ("CREATE TRIGGER accounts_audit AFTER UPDATE ON accounts\nBEGIN",
     "CREATE TRIGGER accounts_audit AFTER UPDATE OF balance_cents ON accounts\n"
     "WHEN OLD.balance_cents <> NEW.balance_cents\nBEGIN"),
    ("    SELECT RAISE(ABORT, 'balance must be zero');\nEND;",
     "    SELECT RAISE(ABORT, 'balance must be zero');\nEND;\n\n"
     "CREATE TRIGGER accounts_reopen BEFORE UPDATE OF closed ON accounts\nWHEN OLD.closed = 1 AND NEW.closed <> 1\n"
     "BEGIN\n    SELECT RAISE(ABORT, 'closed accounts stay closed');\nEND;"),
    ("       SUM(tin.amount_cents)  AS total_in,\n       SUM(tout.amount_cents) AS total_out,\n"
     "       COUNT(*)               AS transfer_count\nFROM accounts a\n"
     "LEFT JOIN transfers tin  ON tin.to_id = a.id\nLEFT JOIN transfers tout ON tout.from_id = a.id\nGROUP BY a.id;",
     "       COALESCE((SELECT SUM(amount_cents) FROM transfers WHERE to_id = a.id), 0)   AS total_in,\n"
     "       COALESCE((SELECT SUM(amount_cents) FROM transfers WHERE from_id = a.id), 0) AS total_out,\n"
     "       (SELECT COUNT(*) FROM transfers WHERE to_id = a.id OR from_id = a.id)       AS transfer_count\n"
     "FROM accounts a;"),
]

_LEDGER_BASE = [
    ("do", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (1, 'ann', 'USD', 1000)"),
    ("do", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (2, 'bob', 'USD', 500)"),
    ("do", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (3, 'cy', 'EUR', 300)"),
    ("do", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (4, 'dee', 'USD', 0)"),
]
_BAL = "SELECT id, balance_cents FROM accounts ORDER BY id"
_TR = "INSERT INTO transfers(from_id, to_id, amount_cents) VALUES (%d, %d, %d)"


def ledger_grade(answer, scale=1.0):
    B = _LEDGER_BASE
    checks = [
        ("a transfer moves the money", B + [
            ("do", _TR % (1, 2, 250)), ("do", _TR % (2, 4, 700)),
            ("rows", _BAL, [(1, 750), (2, 50), (3, 300), (4, 700)]),
            ("rows", "SELECT COUNT(*) FROM transfers", [(2,)])]),
        ("overdraft and unknown accounts are rejected", B + [
            ("fail", _TR % (2, 1, 501)), ("fail", _TR % (1, 99, 10)), ("fail", _TR % (98, 1, 10)),
            ("rows", _BAL, [(1, 1000), (2, 500), (3, 300), (4, 0)]),
            ("rows", "SELECT COUNT(*) FROM transfers", [(0,)])]),
        ("transfers cannot be deleted; closing needs a zero balance", B + [
            ("do", _TR % (1, 2, 100)),
            ("fail", "DELETE FROM transfers"),
            ("fail", "UPDATE accounts SET closed = 1 WHERE id = 1"),
            ("do", "UPDATE accounts SET closed = 1 WHERE id = 4"),
            ("rows", "SELECT id, closed FROM accounts ORDER BY id", [(1, 0), (2, 0), (3, 0), (4, 1)])]),
        ("Q1: one account per owner and currency", B + [
            ("do", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (5, 'ann', 'EUR', 10)"),
            ("fail", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (6, 'ann', 'USD', 10)"),
            ("fail", "INSERT INTO accounts(id, owner, currency, balance_cents) VALUES (7, 'bob', 'USD', 0)")]),
        ("Q1: currency has exactly 3 characters", B + [
            ("fail", "INSERT INTO accounts(id, owner, currency) VALUES (5, 'eve', 'US')"),
            ("fail", "INSERT INTO accounts(id, owner, currency) VALUES (5, 'eve', '')"),
            ("fail", "INSERT INTO accounts(id, owner, currency) VALUES (5, 'eve', 'USDX')"),
            ("do", "INSERT INTO accounts(id, owner, currency) VALUES (5, 'eve', 'BRL')")]),
        ("Q2: amount must be positive", B + [
            ("fail", _TR % (1, 2, 0)), ("fail", _TR % (1, 2, -50)),
            ("rows", _BAL, [(1, 1000), (2, 500), (3, 300), (4, 0)]),
            ("rows", "SELECT COUNT(*) FROM transfers", [(0,)])]),
        ("Q2: no transfer to the same account", B + [
            ("fail", _TR % (1, 1, 10)),
            ("rows", "SELECT COUNT(*) FROM transfers", [(0,)]),
            ("rows", _BAL, [(1, 1000), (2, 500), (3, 300), (4, 0)])]),
        ("Q3: currencies must match", B + [
            ("fail", _TR % (1, 3, 10)), ("fail", _TR % (3, 2, 10)),
            ("rows", _BAL, [(1, 1000), (2, 500), (3, 300), (4, 0)]),
            ("do", _TR % (1, 2, 10))]),
        ("Q3: closed accounts cannot send or receive", B + [
            ("do", "UPDATE accounts SET closed = 1 WHERE id = 4"),
            ("fail", _TR % (1, 4, 10)),
            ("rows", _BAL, [(1, 1000), (2, 500), (3, 300), (4, 0)]),
            ("do", _TR % (2, 1, 500)),
            ("do", "UPDATE accounts SET closed = 1 WHERE id = 2"),
            ("fail", _TR % (2, 1, 0 + 1)),
            ("fail", _TR % (1, 2, 5)),
            ("rows", _BAL, [(1, 1500), (2, 0), (3, 300), (4, 0)])]),
        ("Q4: transfers cannot be updated", B + [
            ("do", _TR % (1, 2, 100)),
            ("fail", "UPDATE transfers SET amount_cents = 1"),
            ("fail", "UPDATE transfers SET to_id = 4"),
            ("fail", "DELETE FROM transfers WHERE from_id = 1"),
            ("rows", "SELECT from_id, to_id, amount_cents FROM transfers", [(1, 2, 100)])]),
        ("Q5: audit rows only for balance changes", B + [
            ("do", _TR % (1, 2, 100)),
            ("do", "UPDATE accounts SET owner = 'ann-marie' WHERE id = 1"),
            ("do", "UPDATE accounts SET closed = 1 WHERE id = 4"),
            ("do", "UPDATE accounts SET balance_cents = 300 WHERE id = 3"),
            ("rows", "SELECT account_id, old_balance, new_balance FROM audit_log ORDER BY account_id",
             [(1, 1000, 900), (2, 500, 600)])]),
        ("Q5: every balance change is logged", B + [
            ("do", _TR % (1, 2, 100)), ("do", _TR % (2, 1, 600)),
            ("do", "UPDATE accounts SET balance_cents = 5 WHERE id = 3"),
            ("rows", "SELECT account_id, old_balance, new_balance FROM audit_log ORDER BY account_id, old_balance",
             [(1, 900, 1500), (1, 1000, 900), (2, 500, 600), (2, 600, 0), (3, 300, 5)])]),
        ("Q6: a closed account cannot be reopened", B + [
            ("do", "UPDATE accounts SET closed = 1 WHERE id = 4"),
            ("fail", "UPDATE accounts SET closed = 0 WHERE id = 4"),
            ("fail", "UPDATE accounts SET closed = 1 WHERE id = 3"),
            ("rows", "SELECT id, closed FROM accounts ORDER BY id", [(1, 0), (2, 0), (3, 0), (4, 1)])]),
        ("Q7: statement view with transfers", B + [
            ("do", _TR % (1, 2, 100)), ("do", _TR % (1, 2, 50)), ("do", _TR % (2, 1, 30)), ("do", _TR % (1, 4, 20)),
            ("rows", "SELECT account_id, owner, total_in, total_out, transfer_count FROM account_statement ORDER BY account_id",
             [(1, "ann", 30, 170, 4), (2, "bob", 150, 30, 3), (3, "cy", 0, 0, 0), (4, "dee", 20, 0, 1)])]),
        ("Q7: statement view without transfers", B + [
            ("rows", "SELECT account_id, owner, total_in, total_out, transfer_count FROM account_statement ORDER BY account_id",
             [(1, "ann", 0, 0, 0), (2, "bob", 0, 0, 0), (3, "cy", 0, 0, 0), (4, "dee", 0, 0, 0)])]),
    ]
    return grade_behaviour(answer, checks)


task(id="sql_bugfix_ledger", lang="sql", category="sql",
     title="Schema and triggers with hidden bugs",
     spec=LEDGER_SPEC, grade=ledger_grade, given=LEDGER_GIVEN, fixes=LEDGER_FIXES, raw_prompt=True)


# ---- SQL 6: query and index optimization -----------------------------------

EVENTS_SCHEMA = [
    "CREATE TABLE users (id INTEGER PRIMARY KEY, country TEXT NOT NULL, signup_date TEXT NOT NULL)",
    "CREATE TABLE events (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, kind TEXT NOT NULL, created_at TEXT NOT NULL, amount_cents INTEGER)",
]

EVENTS_GIVEN = r"""SELECT u.id,
       (SELECT COUNT(*) FROM events e
         WHERE e.user_id = u.id AND e.kind = 'purchase') AS purchases,
       (SELECT SUM(e.amount_cents) FROM events e
         WHERE e.user_id = u.id AND e.kind = 'purchase'
           AND strftime('%Y-%m', e.created_at) = '2026-03') AS march_cents,
       (SELECT MAX(e.created_at) FROM events e WHERE e.user_id = u.id) AS last_event
FROM users u
WHERE lower(u.country) = 'br'
  AND u.id IN (SELECT e.user_id FROM events e WHERE e.kind = 'refund')
ORDER BY purchases DESC, u.id;
"""

EVENTS_SPEC = r"""A SQLite database has these tables and NO indexes other than the primary keys:

  CREATE TABLE users (id INTEGER PRIMARY KEY, country TEXT NOT NULL, signup_date TEXT NOT NULL);
  CREATE TABLE events (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, kind TEXT NOT NULL,
                       created_at TEXT NOT NULL, amount_cents INTEGER);

Facts about the data: country is always two upper-case letters; kind is one of 'view', 'purchase', 'refund'; created_at is 'YYYY-MM-DD HH:MM:SS'; amount_cents can be NULL. The production database has tens of thousands of users and hundreds of thousands of events.

This query is correct but takes minutes in production:

```sql
""" + EVENTS_GIVEN + r"""```

Make it fast. Your answer must be ONE ```sql block containing zero or more CREATE INDEX statements followed by exactly one SELECT statement (it may start with WITH). Nothing else is accepted: you cannot change tables, data or settings.
The SELECT must return exactly the same rows, columns and order as the original query on any data (same NULLs too).
Your script is timed as a whole (index creation plus the query) on databases of increasing size, and it must finish within a few seconds on the largest one."""

EVENTS_SOLUTION = r"""
CREATE INDEX idx_events_user_kind ON events(user_id, kind, created_at, amount_cents);
SELECT u.id,
       COALESCE(a.purchases, 0) AS purchases,
       a.march_cents,
       a.last_event
FROM users u
JOIN (SELECT e.user_id,
             SUM(e.kind = 'purchase') AS purchases,
             SUM(CASE WHEN e.kind = 'purchase' AND e.created_at >= '2026-03-01' AND e.created_at < '2026-04-01'
                      THEN e.amount_cents END) AS march_cents,
             MAX(e.created_at) AS last_event,
             MAX(e.kind = 'refund') AS has_refund
      FROM events e GROUP BY e.user_id) a ON a.user_id = u.id
WHERE u.country = 'BR' AND a.has_refund = 1
ORDER BY purchases DESC, u.id;
"""


def _events_data(seed, nusers, nevents):
    r = Rng(seed)
    countries = ["BR", "BR", "US", "AR", "PT", "DE", "MX", "BR", "CL", "JP"]
    users = [(i, countries[r._step() % 10], "2025-%02d-%02d" % (1 + r._step() % 12, 1 + r._step() % 28))
             for i in range(1, nusers + 1)]
    kinds = ["view", "view", "view", "view", "purchase", "purchase", "purchase", "view", "view", "view",
             "view", "purchase", "view", "view", "view", "view", "view", "refund", "view", "view",
             "view", "view", "view", "view", "view"]
    months = ["2026-01", "2026-02", "2026-03", "2026-03", "2026-04", "2025-03"]
    events = []
    step = r._step
    for i in range(1, nevents + 1):
        x = step()
        y = step()
        kind = kinds[x % 25]
        amount = None if (kind == "view" or y % 13 == 0) else 100 + (y >> 4) % 90000
        ts = "%s-%02d %02d:%02d:%02d" % (months[(x >> 6) % 6], 1 + (x >> 9) % 28, (x >> 14) % 24, (y >> 5) % 60,
                                         (y >> 11) % 60)
        events.append((i, 1 + (x >> 3) % nusers if (y >> 20) % 10 else 1 + (x >> 3) % max(1, nusers // 50),
                       kind, ts, amount))
    return users, events


def _events_expected(users, events):
    agg = {}
    for _, uid, kind, ts, amount in events:
        a = agg.get(uid)
        if a is None:
            a = agg[uid] = [0, None, ts, False]
        if kind == "purchase":
            a[0] += 1
            if ts[:7] == "2026-03" and amount is not None:
                a[1] = (a[1] or 0) + amount
        elif kind == "refund":
            a[3] = True
        if ts > a[2]:
            a[2] = ts
    rows = [(uid, agg[uid][0], agg[uid][1], agg[uid][2]) for uid, country, _ in users
            if country == "BR" and uid in agg and agg[uid][3]]
    rows.sort(key=lambda t: (-t[1], t[0]))
    return rows


def events_grade(answer, scale=1.0):
    stmts = sql_statements(answer)
    bad = None
    if not stmts:
        bad = "no SQL found"
    else:
        for s in stmts[:-1]:
            if not re.match(r"CREATE (UNIQUE )?INDEX\b", sql_head(s)):
                bad = "only CREATE INDEX statements may precede the SELECT (found: %s)" % shorten(s, 60)
        head = sql_head(stmts[-1])
        if not (head.startswith("SELECT") or head.startswith("WITH")):
            bad = "the last statement must be the SELECT"
    plan = [
        ("small data set 1", 3001, 60, 700, 20.0, 0.5, False),
        ("small data set 2 (users without purchases, NULL amounts)", 3002, 25, 150, 20.0, 0.5, False),
        ("small data set 3", 3003, 200, 2500, 20.0, 0.5, False),
        ("tier 1: 10000 users, 160000 events", 3004, 10000, 160000, 3.0, 2.5, True),
        ("tier 2: 15000 users, 250000 events", 3005, 15000, 250000, 3.0, 2.5, True),
        ("tier 3: 30000 users, 600000 events", 3006, 30000, 600000, 4.0, 3.0, True),
    ]
    results = []
    for name, seed, nu, ne, limit, weight, timed in plan:
        if bad:
            results.append(R(name, False, bad, weight))
            continue
        users, events = _events_data(seed, nu, ne)
        want = norm_rows(_events_expected(users, events))
        conn = sql_connect()
        try:
            for s in EVENTS_SCHEMA:
                conn.execute(s)
            conn.execute("BEGIN")
            conn.executemany("INSERT INTO users VALUES (?, ?, ?)", users)
            conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", events)
            conn.execute("COMMIT")
            lim = limit * (scale if timed else 1.0)
            sql_guard(conn, lim)
            t0 = time.monotonic()
            try:
                for s in stmts[:-1]:
                    conn.execute(s)
                got = sql_fetch(conn, stmts[-1])
            except Exception as e:
                secs = time.monotonic() - t0
                msg = "time limit of %.1fs exceeded" % lim if "interrupt" in str(e).lower() else \
                    "failed: %s" % shorten(str(e), 200)
                res = R(name, False, msg, weight)
                res["secs"] = secs
                results.append(res)
                continue
            secs = time.monotonic() - t0
            ok = got == want
            res = R(name, ok, "" if ok else "wrong result: got %d rows %s, expected %d rows %s" % (
                len(got), shorten(repr(got[:3]), 160), len(want), shorten(repr(want[:3]), 160)), weight)
            res["secs"] = secs
            results.append(res)
        finally:
            conn.close()
    return results


task(id="sql_opt_events", lang="sql", category="sql",
     title="Query and index optimization",
     spec=EVENTS_SPEC, grade=events_grade, given=EVENTS_GIVEN, solution=EVENTS_SOLUTION, raw_prompt=True)


# ==========================================================================
# Prompts
# ==========================================================================

BUGFIX_INTRO = (
    "The program below is supposed to implement the following specification, but it contains "
    "bugs. Their number is not disclosed; some of them only show up with particular inputs. "
    "Find and fix ALL of them without changing the input and output formats, and return the "
    "complete corrected program.")

OPTIMIZE_INTRO = (
    "The program below is correct but far too slow: it must handle the largest inputs allowed "
    "by the specification in about one second. Rewrite it so that it is as fast as possible "
    "while printing exactly the same output for every valid input, and return the complete "
    "program.")


def build_prompt(t, tc):
    if t.get("raw_prompt"):
        return t["spec"]
    lang = t["lang"]
    tail = "%s\n%s" % (lang_requirements(lang, tc), RESPONSE_RULES)
    if t["category"] == "write":
        return "%s\n\n%s" % (t["spec"], tail)
    intro = BUGFIX_INTRO if t["category"] == "bugfix" else OPTIMIZE_INTRO
    return "%s\n\nSPECIFICATION\n%s\n\nPROGRAM\n```%s\n%s```\n\n%s" % (
        intro, t["spec"], LANG_FENCES[lang][0], t["given"], tail)


# ==========================================================================
# Grading
# ==========================================================================

NATIVE_MEM = 3 << 30          # address-space limit for Python, PHP and non-sanitized C/C++
CORRECTNESS_LIMIT = 6.0       # seconds for non-timed tests (x2 under sanitizers)


def first_diff(got, want):
    g, w = got.split("\n"), want.split("\n")
    for i in range(max(len(g), len(w))):
        a = g[i] if i < len(g) else "<missing>"
        b = w[i] if i < len(w) else "<nothing>"
        if a != b:
            return "line %d: got %s, expected %s" % (i + 1, shorten(repr(a), 90), shorten(repr(b), 90))
    return "outputs differ"


def grade_program(t, code, tc, scale, workdir):
    """Compile and run one program against the task's hidden tests."""
    lang = t["lang"]
    built = build_program(lang, code, workdir, tc)
    tests = t["tests"]()
    info = {"compiled": built.ok, "compile_log": shorten(built.log, 1500) if not built.ok else "",
            "warnings": built.warnings, "sanitizers": bool(built.cmd_san), "tests": []}
    if not built.ok:
        info["tests"] = [R(x["name"], False, "does not compile", x["weight"]) for x in tests]
        return info
    slow_timeouts = 0
    for x in tests:
        data = x["input"]
        if isinstance(data, str):
            data = data.encode("utf-8")
        use_san = (not x["timed"]) and built.cmd_san is not None
        cmd = built.cmd_san if use_san else built.cmd
        if x["timed"]:
            limit = x["limit"] * scale + STARTUP_ALLOWANCE[lang]
        else:
            limit = CORRECTNESS_LIMIT * max(scale, 1.0) * (2 if use_san else 1) + STARTUP_ALLOWANCE[lang]
            if slow_timeouts >= 3:
                info["tests"].append(R(x["name"], False, "not run: the program already timed out on 3 "
                                       "simple tests", x["weight"]))
                continue
        mem = NATIVE_MEM if (lang in ("python", "php") or (lang in ("c", "cpp") and not use_san)) else None
        for attempt in range(SAN_RETRIES if use_san else 1):
            verdict, out, err, secs = run_case(cmd, workdir, data, limit, mem,
                                               fsize_limit=lang not in ("csharp", "java"),
                                               no_aslr=use_san)
            # A real failure reproduces or carries a sanitizer report; a start-up failure does not.
            if verdict != "crash" or has_sanitizer_report(err):
                break
        if verdict == "timeout":
            res = R(x["name"], False, "time limit of %.1fs exceeded" % limit, x["weight"])
            if not x["timed"]:
                slow_timeouts += 1
        elif verdict == "crash":
            reason = "runtime error"
            if "Sanitizer" in err:
                m = re.search(r"(AddressSanitizer|UndefinedBehaviorSanitizer|LeakSanitizer)[^\n]*", err)
                reason = "memory/UB error: " + (m.group(0) if m else "sanitizer report")
            elif "runtime error:" in err:
                m = re.search(r"runtime error:[^\n]*", err)
                reason = "undefined behaviour: " + m.group(0)
            else:
                lines = [ln for ln in err.strip().splitlines() if ln.strip()]
                if lines:
                    reason += ": " + " | ".join(lines[-2:])
            res = R(x["name"], False, shorten(reason, 240), x["weight"])
        else:
            got, want = norm_out(out), norm_out(x["expected"])
            if got == want:
                res = R(x["name"], True, "", x["weight"])
            else:
                res = R(x["name"], False, "wrong answer, " + first_diff(got, want), x["weight"])
        res["secs"] = round(secs, 3)
        info["tests"].append(res)
    return info


def grade_sql(t, answer_sql, scale):
    results = t["grade"](answer_sql, scale)
    return {"compiled": True, "compile_log": "", "warnings": 0, "sanitizers": False, "tests": results}


def request_failed(rec):
    """True when a stored record holds no real answer (request error or broken stream)."""
    gen = rec.get("gen") or {}
    return bool(gen.get("error")) or gen.get("finish") is None


def score_of(tests):
    total = sum(x["weight"] for x in tests)
    got = sum(x["weight"] for x in tests if x["passed"])
    return (got / total) if total else 0.0


def grade_answer(t, answer, tc, scale, keep_dir=None):
    """Extract the code from a chat answer and grade it. Returns a result dict."""
    lang = t["lang"]
    code, how = extract_code(answer, lang)
    if lang == "sql":
        info = grade_sql(t, code, scale)
    else:
        workdir = tempfile.mkdtemp(prefix="cb_%s_" % t["id"])
        try:
            info = grade_program(t, code, tc, scale, workdir)
        finally:
            if keep_dir:
                shutil.rmtree(keep_dir, ignore_errors=True)
                shutil.copytree(workdir, keep_dir, ignore=shutil.ignore_patterns("_in.txt", "_out.txt"))
            shutil.rmtree(workdir, ignore_errors=True)
    info["extraction"] = how
    info["code"] = code
    info["score"] = score_of(info["tests"])
    info["passed"] = sum(1 for x in info["tests"] if x["passed"])
    info["total"] = len(info["tests"])
    return info


# ==========================================================================
# Self-test (no model needed)
# ==========================================================================

def task_reference(t):
    """Known-good answer for a task (used by --selftest only)."""
    if "fixes" in t:
        return apply_fixes(t["given"], t["fixes"])
    sol = t.get("solution")
    return sol() if callable(sol) else sol


def selftest_extraction():
    """extract_code on answers with missing, stray or unusual fences. Returns failing case names."""
    f = "```"
    code = "import sys\nn = int(sys.stdin.readline())\nfor i in range(n):\n    print(i)\n"
    prose = "This solution reads the input and prints each number.\n"
    cases = [
        ("both fences", "Here is the solution.\n\n%spython\n%s%s\nDone.\n" % (f, code, f)),
        ("no fence", code),
        ("closing fence missing", "Here it is:\n%spython\n%s" % (f, code)),
        ("opening fence missing", code + f + "\n"),
        ("opening fence missing, prose after", "Here is the solution:\n\n" + code + f + "\n" + prose),
        ("opening fence missing, block after", code + f + "\nExample:\n%s\n3 4\n%s\n" % (f, f)),
        ("CRLF, closing fence missing", (f + "python\n" + code).replace("\n", "\r\n")),
        ("backticks inside a sentence", "Use %s fences.\n%spython\n%s%s\n" % (f, f, code, f)),
        ("first block never closed", "%spython\nx = 1\n%spython\n%s%s\n" % (f, f, code, f)),
    ]
    return [name for name, answer in cases if extract_code(answer, "python")[0] != code]


def selftest(tasks, tc, scale):
    print("Self-test: every reference solution must score 100%; every original (buggy/slow) program "
          "must score low.\n")
    bad = selftest_extraction()
    print("Code extraction: %s\n" % ("FAILED (%s)" % ", ".join(bad) if bad else "ok"))
    print("%-22s %-7s %10s %10s  %s" % ("task", "lang", "reference", "original", "notes"))
    ok_all = not bad
    for t in tasks:
        lang = t["lang"]
        if not tc[lang]["ok"]:
            print("%-22s %-7s %10s %10s  toolchain missing" % (t["id"], lang, "-", "-"))
            continue
        fence = LANG_FENCES[lang][0]
        t0 = time.monotonic()
        ref = task_reference(t)
        res = grade_answer(t, "Here is the solution.\n\n```%s\n%s```\nDone.\n" % (fence, ref), tc, scale)
        notes = []
        ref_txt = "%.1f%%" % (100 * res["score"])
        if res["score"] < 1.0:
            ok_all = False
            bad = [x for x in res["tests"] if not x["passed"]]
            notes.append("REFERENCE FAILS: " + "; ".join("%s (%s)" % (x["name"], x["detail"]) for x in bad[:3]))
            if not res["compiled"]:
                notes.append(shorten(res["compile_log"], 300))
        slowest = max([x.get("secs", 0) for x in res["tests"]] or [0])
        orig_txt = "-"
        if "given" in t:
            res2 = grade_answer(t, "```%s\n%s```\n" % (fence, t["given"]), tc, scale)
            orig_txt = "%.1f%%" % (100 * res2["score"])
            if res2["score"] >= 0.5:
                ok_all = False
                notes.append("ORIGINAL SCORES TOO HIGH")
        notes.append("slowest test %.2fs, total %.0fs" % (slowest, time.monotonic() - t0))
        if lang in ("c", "cpp"):
            notes.append("sanitizers on" if res["sanitizers"] else "sanitizers OFF")
        print("%-22s %-7s %10s %10s  %s" % (t["id"], lang, ref_txt, orig_txt, " | ".join(notes)))
        sys.stdout.flush()
    print("\nSelf-test %s." % ("PASSED" if ok_all else "FAILED"))
    return 0 if ok_all else 1


# ==========================================================================
# Reports
# ==========================================================================

def slugify(name):
    name = os.path.basename(name.strip().rstrip("/"))
    name = re.sub(r"\.gguf$", "", name, flags=re.I)
    name = re.sub(r"[^A-Za-z0-9._+-]+", "-", name).strip("-.")
    return name or "model"


def machine_info():
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for ln in f:
                if ln.lower().startswith("model name"):
                    cpu = ln.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    import platform
    return "%s | %s" % (cpu or platform.processor() or "unknown CPU", platform.platform())


def aggregate(results, tasks):
    """results: list of per-(task, run) records. Returns summary structures."""
    by_task = {}
    for rec in results:
        by_task.setdefault(rec["task"], []).append(rec)
    task_rows = []
    for t in tasks:
        recs = by_task.get(t["id"])
        if not recs:
            continue
        scores = [r["score"] for r in recs]
        task_rows.append({
            "id": t["id"], "title": t["title"], "lang": t["lang"], "category": t["category"],
            "score": sum(scores) / len(scores), "scores": scores,
            "passed": sum(r["passed"] for r in recs) / len(recs), "total": recs[0]["total"],
            "compiled": sum(1 for r in recs if r["compiled"]) / len(recs),
            "gen_seconds": sum(r["gen"]["seconds"] for r in recs) / len(recs),
            "tokens": sum((r["gen"]["completion_tokens"] or 0) for r in recs) / len(recs),
            "tps": statistics.mean([r["gen"]["tps"] for r in recs if r["gen"]["tps"]] or [0]),
            "finish": ",".join(sorted(set(str(r["gen"]["finish"]) for r in recs))),
            "errors": [r["gen"]["error"] for r in recs if r["gen"]["error"]],
        })
    runs = sorted(set(r["run"] for r in results))
    per_run = []
    ids = set(r["id"] for r in task_rows)  # stored records of removed tasks do not count
    for k in runs:
        sc = [r["score"] for r in results if r["run"] == k and r["task"] in ids]
        if len(sc) == len(task_rows):
            per_run.append(100.0 * sum(sc) / len(sc))
    final = 100.0 * sum(r["score"] for r in task_rows) / len(task_rows) if task_rows else 0.0

    def group(keyfn, labels=None):
        groups = {}
        for r in task_rows:
            groups.setdefault(keyfn(r), []).append(r["score"])
        return [(labels.get(k, k) if labels else k, 100.0 * sum(v) / len(v), len(v)) for k, v in groups.items()]

    tps_mean, tps_stdev = tps_stats(results)
    return {
        "final": final, "per_run": per_run,
        "stdev": statistics.pstdev(per_run) if len(per_run) > 1 else None,
        "tasks": task_rows,
        "by_category": group(lambda r: r["category"], CATEGORY_LABEL),
        "by_language": group(lambda r: LANG_GROUP[r["lang"]]),
        "perfect": sum(1 for r in task_rows if r["score"] >= 0.9999),
        "compile_rate": 100.0 * sum(r["compiled"] for r in task_rows) / len(task_rows) if task_rows else 0.0,
        "gen_seconds": sum(r["gen"]["seconds"] for r in results),
        "tokens": sum((r["gen"]["completion_tokens"] or 0) for r in results),
        "tps": tps_mean, "tps_stdev": tps_stdev,
        "truncated": sum(1 for r in results if r["gen"]["finish"] in CUT_OFF),
        "gen_errors": sum(1 for r in results if r["gen"]["error"]),
    }


def tps_stats(results):
    """Mean and standard deviation of the generation speed over all answers."""
    v = [r["gen"]["tps"] for r in results if r["gen"]["tps"]]
    if not v:
        return 0.0, None
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else None)


def fmt_tps(mean, stdev):
    return "%.1f" % mean + ("" if stdev is None else " ±%.1f" % stdev)


def beta_number(version):
    m = re.match(r"1\.0 Beta (\d+)$", version or "")
    return int(m.group(1)) if m else None


def version_group(version):
    """Versions listed in COMPAT share one group; any other version is a group of its own."""
    m = re.match(r"1\.0 Beta (\d+)$", version or "")
    return "compat" if m and int(m.group(1)) in COMPAT else version


def fmt_secs(s):
    s = int(round(s))
    if s >= 3600:
        return "%dh%02dm" % (s // 3600, s % 3600 // 60)
    if s >= 60:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%ds" % s


def write_report(path, meta, results, tasks):
    agg = aggregate(results, tasks)
    L = []
    add = L.append
    add("# Coding benchmark report: %s" % meta["name"])
    add("")
    add("## Final score: %.1f / 100" % agg["final"])
    add("")
    if agg["stdev"] is not None:
        add("Mean of %d runs (per run: %s; standard deviation %.1f)." % (
            len(agg["per_run"]), ", ".join("%.1f" % x for x in agg["per_run"]), agg["stdev"]))
        add("")
    if meta.get("partial"):
        add("**PARTIAL RESULT: %d of %d task runs were completed.**" % (len(results), meta["planned"]))
        add("")
    if meta.get("skipped_langs"):
        add("**Not comparable with full runs: tasks skipped for missing toolchains (%s).**" %
            ", ".join(meta["skipped_langs"]))
        add("")
    add("The score is the average, over %d tasks, of the fraction of hidden tests each answer passed "
        "(weighted inside each task). Every test is checked by a program; nothing is judged by eye." %
        len(agg["tasks"]))
    add("")
    add("| Item | Value |")
    add("|---|---|")
    add("| Date | %s |" % meta["date"])
    add("| Model (server id) | %s |" % (meta["server"].get("model_id") or "?"))
    if meta["server"].get("model_path"):
        add("| Model path | %s |" % meta["server"]["model_path"])
    add("| Endpoint | %s |" % meta["url"])
    if meta["server"].get("n_ctx"):
        add("| Context size | %s |" % meta["server"]["n_ctx"])
    if meta["server"].get("sampling"):
        add("| Server sampling defaults | %s |" % ", ".join(
            "%s=%s" % kv for kv in sorted(meta["server"]["sampling"].items())))
    if meta["server"].get("build_info"):
        add("| llama.cpp build | %s |" % meta["server"]["build_info"])
    add("| Request overrides | %s |" % (json.dumps(meta["overrides"]) if meta["overrides"] else "none (server defaults)"))
    add("| Runs per task / base seed | %d / %d |" % (meta["runs"], meta["seed"]))
    add("| Max tokens / time per answer | %d / %s |" % (meta["max_tokens"], fmt_secs(meta["task_timeout"])))
    add("| Time-limit scale | %.2f |" % meta["scale"])
    add("| Machine | %s |" % meta["machine"])
    add("| Toolchains | %s |" % "; ".join("%s: %s" % (LANG_LABEL[k], v) for k, v in meta["toolchains"].items()))
    add("| Benchmark version | %s |" % meta.get("version", "unknown"))
    mig = meta.get("migrated_from")
    if mig:
        add("| Migrated from | %s (%s); asked again: %s |" % (
            mig.get("version", "?"), mig.get("dir", "?"), ", ".join(mig.get("redo", []))))
    add("")

    add("## Score by category")
    add("")
    add("| Category | Score | Tasks |")
    add("|---|---:|---:|")
    for label, sc, n in agg["by_category"]:
        add("| %s | %.1f | %d |" % (label, sc, n))
    add("")
    add("## Score by language")
    add("")
    add("| Language | Score | Tasks |")
    add("|---|---:|---:|")
    for label, sc, n in agg["by_language"]:
        add("| %s | %.1f | %d |" % (label, sc, n))
    add("")

    add("## Tasks")
    add("")
    add("| Task | Language | Category | Score | Tests passed | Builds | Answer time | Tokens | tok/s | Finish |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in agg["tasks"]:
        add("| %s | %s | %s | %.1f | %s/%d | %d%% | %s | %d | %.1f | %s |" % (
            r["id"], LANG_LABEL[r["lang"]], r["category"], 100 * r["score"],
            ("%.1f" % r["passed"]).rstrip("0").rstrip("."), r["total"], round(100 * r["compiled"]),
            fmt_secs(r["gen_seconds"]), round(r["tokens"]), r["tps"], r["finish"]))
    add("")

    add("## Secondary metrics (tie-breakers, not part of the score)")
    add("")
    add("| Metric | Value |")
    add("|---|---:|")
    add("| Tasks solved completely | %d of %d |" % (agg["perfect"], len(agg["tasks"])))
    add("| Answers that compile/parse at the first attempt | %.0f%% |" % agg["compile_rate"])
    add("| Answers cut off (token limit, time limit or loop) | %d |" % agg["truncated"])
    add("| Failed requests | %d |" % agg["gen_errors"])
    add("| Total generation time | %s |" % fmt_secs(agg["gen_seconds"]))
    add("| Total generated tokens | %d |" % agg["tokens"])
    add("| Average speed (± standard deviation over the answers) | %s tok/s |"
        % fmt_tps(agg["tps"], agg["tps_stdev"]))
    add("")

    add("## Failed tests")
    add("")
    any_fail = False
    for rec in results:
        failed = [x for x in rec["tests"] if not x["passed"]]
        if not failed and not rec["gen"]["error"]:
            continue
        any_fail = True
        add("### %s (run %d): %.1f" % (rec["task"], rec["run"], 100 * rec["score"]))
        add("")
        if rec["gen"]["error"]:
            add("- Request error: %s" % rec["gen"]["error"])
        if rec["gen"]["finish"] in CUT_OFF:
            add("- The answer was cut off (%s)." % rec["gen"]["finish"])
        add("- Code taken from: %s" % rec["extraction"])
        if not rec["compiled"]:
            add("- Does not compile:")
            add("")
            add("```")
            add(rec["compile_log"].replace("```", "'''"))
            add("```")
        else:
            for x in failed[:8]:
                add("- %s: %s" % (x["name"], x["detail"].replace("\n", " ")))
            if len(failed) > 8:
                add("- ... and %d more" % (len(failed) - 8))
        add("")
    if not any_fail:
        add("None.")
        add("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return agg


def cutoff_stats(results):
    """Extra figures for --compare: cut-off answers by reason, and the `stop` answers alone."""
    kinds = {k: sum(1 for r in results if r["gen"]["finish"] == k) for k in CUT_OFF}
    done = [r for r in results if r["gen"]["finish"] == "stop"]
    return {
        "total": len(results),
        "cut": sum(kinds.values()),
        "kinds": kinds,
        "stop_n": len(done),
        "stop_score": 100 * sum(r["score"] for r in done) / len(done) if done else None,
        "stop_tokens": statistics.median(r["gen"]["completion_tokens"] for r in done) if done else None,
    }


# ==========================================================================
# HTML report (comparison.html)
# ==========================================================================

# favicon.png (32x32), embedded so that the HTML page is a single self-contained file.
FAVICON_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAACXBIWXMAAAsTAAALEwEAmpwYAAAA"
    "AXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAMAUExURUdwTKamppWVlYKCgpGRkYaGhouLi4uL"
    "i3d3d2pqanp6enV1dVRUVElJSHFxcXJyclpaWl1dXX5+fnt7e1VVVZKSkoODg4GBgXt7e4eHh4CA"
    "gH19fX9/f2NjY1lZWUtLS3Nzc4GBgYGBgVVVVXh4eGhoaGhoaGNjY4qKioWFhX19fVpaWmdnZ1dX"
    "V2xsbG9vb3x8fHl5eY6OjpGRkXBwcH19fXh4eE1NTX5+fnNzc1RUVFFRUWNjY0ZGRlRUVFNTU2Zm"
    "ZnBwcGdnZ3FxcXNzc3R0dHp6en19fVlZWf763P352/z42vj01/v32vn12Pr22Xt7e9HOtfHu0vTw"
    "1OzpzX59fP353E5OTWdnZkxMS9bTunV1dOThxnp6eV5eW1lZV1NTUcfDre/rz+PfxGpqZ9/bwfby"
    "1trYvkdHRrq3o4+OhND/OoeGf5iWhlNTU7e1oc/Msz09PczJsqOgjufjx/v/OvXx1Pfz1razn42L"
    "f7+8purmy+5QK25uboaEeOznvtDSsunjtq/0RrP/ONjavpX/N+nlycnTspWUidLSp9jVu3zWanr/"
    "Nlr+NcnGr19eV4F/c7Gum6SiknFvZHh3bujkyXx8d+v/OubiyN3av/v2OuLewmRkYtG9qtnWvGBg"
    "XldXVNPQtygoKPnjN4mIeIqJgMC9p8nHsPRTGuXixjEwLnJxbPRfG1hYU1JRTrWynJqYifNsHfDt"
    "0YSEfejny8z7PqPHkpHnV621nuT8Pa/LnILbYpflWWfjVJPVdPX8PtvdwWb/Nej/OsnRoInTeIjS"
    "eGjuTHXXZm7fXWTqUF30Q2r9ObXRnmtqYjo6NZWTgby5pp6dj4KAcc3Wc05NRcl7bvzxN/BgJm9u"
    "YFxFM5V6TM2/h/rXMfzrNtJ0ZY5MG5iDVN63SPBrHN1tG/bNM6xyVe7tTcqgj8K/qa5bGtapS/ve"
    "M9PPrZ6cjfTAM3RKLEpJQD09Na+tm4puTOpzHfPGOtJ0ZL+YXMhpG4JVOtPCrdC7p/363O9ZKcOC"
    "dcubgTseQssAAABJdFJOUwABA5sTAy4v/vzhLuH+Lv6bixPxyhmLzPxd6+v8i8rxOxMD8S/8m5s7"
    "Gcrr6/zrzF3KGQPri5vrA/HM6138/OHMXcxdm5sZGZshmOtmAAAC6ElEQVQ4y2NggAEmUUlmcSkZ"
    "GSlxZklRJgZ0wMTOLC0REBAKBAEBEtLM7GhKWMSEYwNyLub3BQb2ZU5aGhArLMaCJM3IpsURm1UV"
    "k+btBQTe5VMTO2I5tNgY4fIaXByhFdW+nnDgnVS1mYNLA6aCjSs6p8DHEwX4R2VFc7FB7TeMzuny"
    "RZX3DM5MzYo2BLuDyZR/WiCSfJiXl6fnn771vumb+U1BfmG35M+Hmu/lH9LS0NnZ+SYqtW52mG9i"
    "tCU7A4OxWW1lCFinz4Ez+/Y0bd25f9bir99//v9VHbSh1syYwcik9oU3UD6t/cix0z2tzdmHSme9"
    "fvf5x8a//9ICV5sYMRhwTmvz9PRtOXqqt/cgSMH00mt3Xr3/sqbmt9empZwGDC4Rb308g3ftPXni"
    "eHf37u1NQAV3n9x6uXHFt02evp8iXBn0IzK9vXdsa4/Z5OPvX7Yl/fDlZ4sf3ny8pmaul6dXUYo+"
    "g15EkZennzfcl17lM28/uP9oxb1NQM7MFD0GzogotDDynBH+9DrIAE/PmBROBtU5XWjyXhNvxDy/"
    "ugmsNEWVQWd+IJqCpP71Xj5TQQZ4zp6jw+AQt8wLotELasCFKyUwduB8Jwa1+GJ/sHhjPTSazp73"
    "a6wHq/CuiFdjcNSdkgRWkNoGiavEc4Fes0vA7LJKXWcGTe34Ai8kFwZPmRwM48ycr63JwKqekAUX"
    "8fRb3pEwEaberzhPnZWBQUV5QSIsnMKK4vLycv2ghkXFKasA0wOrbPjkZKgm7+LCwsL+TRBOyIRw"
    "WVZQklKUC6+rhnq1atWiRZcgyWdTbricIiRRcvNELlkJUZG+6uOSZLDSkNxwHm5Yslew510NTtZe"
    "yz5UgF3g3zUhkkcBkTH4RHgTNqSneXmWzUsCRejsSfG8ItyMSHlLSFA+I6GuYuHKkJCSqMzKBRny"
    "gkKomVOJT8DcNiNy3dq16yIzbM0F+JQw8reFm42HlbWdnbWVgI27BVwYALcjQMmwuPZcAAAAAElF"
    "TkSuQmCC"
)

# GitHub callouts: "> [!NOTE]" (also TIP, IMPORTANT, WARNING, CAUTION): (label, 16x16 icon path).
CALLOUTS = {
    "NOTE": ("Note", "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-2a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"),
    "TIP": ("Tip", "M8 1.5c-2.363 0-4 1.69-4 3.75 0 .984.424 1.625.984 2.304l.214.253c.223.264.47.556.673.848.284.411.537.896.621 1.49a.75.75 0 0 1-1.484.211c-.04-.282-.163-.547-.37-.847a8.456 8.456 0 0 0-.542-.68c-.084-.1-.173-.205-.268-.32C3.201 7.75 2.5 6.766 2.5 5.25 2.5 2.31 4.863 0 8 0s5.5 2.31 5.5 5.25c0 1.516-.701 2.5-1.328 3.259-.095.115-.184.22-.268.319-.207.245-.383.453-.541.681-.208.3-.33.565-.37.847a.751.751 0 0 1-1.485-.212c.084-.593.337-1.078.621-1.489.203-.292.45-.584.673-.848.075-.088.147-.173.213-.253.561-.679.985-1.32.985-2.304 0-2.06-1.637-3.75-4-3.75ZM5.75 12h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1 0-1.5ZM6 15.25a.75.75 0 0 1 .75-.75h2.5a.75.75 0 0 1 0 1.5h-2.5a.75.75 0 0 1-.75-.75Z"),
    "IMPORTANT": ("Important", "M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v9.5A1.75 1.75 0 0 1 14.25 13H8.06l-2.573 2.573A1.458 1.458 0 0 1 3 14.543V13H1.75A1.75 1.75 0 0 1 0 11.25Zm1.75-.25a.25.25 0 0 0-.25.25v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25Zm7 2.25v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 9a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"),
    "WARNING": ("Warning", "M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"),
    "CAUTION": ("Caution", "M4.47.22A.749.749 0 0 1 5 0h6c.199 0 .389.079.53.22l4.25 4.25c.141.14.22.331.22.53v6a.749.749 0 0 1-.22.53l-4.25 4.25A.749.749 0 0 1 11 16H5a.749.749 0 0 1-.53-.22L.22 11.53A.749.749 0 0 1 0 11V5c0-.199.079-.389.22-.53Zm.84 1.28L1.5 5.31v5.38l3.81 3.81h5.38l3.81-3.81V5.31L10.69 1.5ZM8 4a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 4Zm0 8a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"),
}
CALLOUT_MARKER = re.compile(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$", re.I)
HTML_NUM = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:\u00b1.*)?$")

HTML_CSS = """
:root{
  --bg:#f6f5f1; --surface:#ffffff; --ink:#1d2330; --muted:#667085;
  --line:#e4e2da; --head:#1f2a44; --head-ink:#f4f1e8; --accent:#2f6f5e;
  --heat:47,111,94; --gold:#b8892b; --silver:#8a93a3; --bronze:#a8683a;
  --zebra:rgba(31,42,68,.025); --shadow:0 1px 2px rgba(20,26,40,.06),0 8px 28px rgba(20,26,40,.06);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#12151c; --surface:#1a1f2a; --ink:#e8e6df; --muted:#98a1b3;
    --line:#2b3242; --head:#0e1118; --head-ink:#e8e6df; --accent:#6fc2a8;
    --heat:111,194,168; --zebra:rgba(255,255,255,.02);
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.35);
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; padding:56px clamp(16px,4vw,56px) 80px; background:var(--bg); color:var(--ink);
  font:15px/1.6 "Inter","Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",sans-serif;
  font-feature-settings:"tnum" 1,"cv11" 1;
}
.page{max-width:1480px; margin:0 auto}
h1{
  font:600 clamp(28px,4vw,40px)/1.15 "Iowan Old Style","Palatino Linotype",Georgia,serif;
  letter-spacing:-.015em; margin:0 0 6px;
}
h1::after{content:""; display:block; width:56px; height:3px; margin-top:14px; background:var(--accent); border-radius:2px}
h2{
  font:600 22px/1.25 "Iowan Old Style","Palatino Linotype",Georgia,serif;
  letter-spacing:-.01em; margin:56px 0 6px;
}
h1 + p{color:var(--muted); margin:14px 0 28px}
p{margin:.4em 0 1em}
p + .table-wrap, h2 + p + .table-wrap{margin-top:14px}
code{
  font:12.5px ui-monospace,"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
  background:rgba(127,127,127,.14); padding:1px 6px; border-radius:5px;
}
blockquote{
  margin:14px 0 0; padding:10px 16px; border-left:3px solid var(--accent);
  background:var(--surface); border-radius:0 8px 8px 0; color:var(--muted); font-size:13.5px;
}
blockquote p{margin:0}

.callout{
  --c:var(--accent); margin:18px 0; padding:12px 18px 12px 16px; border-left:4px solid var(--c);
  background:color-mix(in srgb,var(--c) 8%,var(--surface)); border-radius:0 10px 10px 0;
  color:var(--ink); font-size:14.5px;
}
.callout p{margin:.35em 0}
.callout-title{
  display:flex; align-items:center; gap:8px; margin:0 0 4px !important; color:var(--c);
  font-weight:600; font-size:13px; letter-spacing:.02em;
}
.callout.note{--c:#2f6fd0} .callout.tip{--c:#2f8f5b} .callout.important{--c:#8250df}
.callout.warning{--c:#b7791f} .callout.caution{--c:#d1383d}
@media (prefers-color-scheme:dark){
  .callout.note{--c:#6ea8ff} .callout.tip{--c:#56d391} .callout.important{--c:#b392f0}
  .callout.warning{--c:#e3b341} .callout.caution{--c:#ff7b72}
}
.table-wrap{
  overflow-x:auto; background:var(--surface); border:1px solid var(--line);
  border-radius:14px; box-shadow:var(--shadow);
}
table{border-collapse:separate; border-spacing:0; width:100%; font-size:13.5px; table-layout:auto}
th,td{padding:9px 12px; white-space:nowrap; border-bottom:1px solid var(--line)}
thead th{
  position:sticky; top:0; z-index:2; background:var(--head); color:var(--head-ink);
  font-weight:600; font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
  padding:13px 12px; border-bottom:0;
}
thead th:first-child{border-top-left-radius:13px}
thead th:last-child{border-top-right-radius:13px}
tbody tr:last-child td{border-bottom:0}
tbody tr:nth-child(even) td{background-color:var(--zebra)}
tbody tr{transition:background .15s}
tbody tr:hover td{background-color:rgba(var(--heat),.09)}
td[style*="center"],th[style*="center"]{text-align:center}
td{font-variant-numeric:tabular-nums}

td.rank{color:var(--muted); font-weight:600; width:1%}
td.model{
  font-weight:600; max-width:420px; overflow:hidden; text-overflow:ellipsis;
  font-family:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace; font-size:12.5px;
}
table.tasks td.task{
  font-family:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace; font-size:12.5px;
  font-weight:500; position:sticky; left:0; background:var(--surface); z-index:1;
  border-right:1px solid var(--line);
}
table.tasks tbody tr:nth-child(even) td.task{background:var(--surface)}
table.tasks thead th:first-child{position:sticky; left:0; z-index:3}

td.heat{
  background-image:linear-gradient(rgba(var(--heat),calc(var(--h)*.42)),rgba(var(--heat),calc(var(--h)*.42)));
}
tbody tr:hover td.heat{background-color:transparent}

tr.podium td.rank{color:var(--ink)}
tr.p1 td.rank{box-shadow:inset 3px 0 0 var(--gold); color:var(--gold)}
tr.p2 td.rank{box-shadow:inset 3px 0 0 var(--silver); color:var(--silver)}
tr.p3 td.rank{box-shadow:inset 3px 0 0 var(--bronze); color:var(--bronze)}
table.ranking td:nth-child(3){font-weight:700}

@media (max-width:640px){
  body{padding-top:32px}
  th,td{padding:8px 9px}
}
@page{size:A4 landscape; margin:12mm}
@media print{
  body{background:#fff; color:#000; padding:0; font-size:11px}
  .table-wrap{box-shadow:none; overflow:visible; break-inside:avoid}
  table{font-size:9px}
  th,td{padding:4px 5px}
  thead th{position:static; background:#222; -webkit-print-color-adjust:exact; print-color-adjust:exact}
  td.heat{-webkit-print-color-adjust:exact; print-color-adjust:exact}
  h2{break-after:avoid}
}
"""


def md_inline(text):
    """Escape text and render the only inline Markdown that comparison.md uses."""
    text = html_escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def _table_cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _heat(pct):
    """Tint of a cell, proportional to a 0-100 value (driven by a CSS variable)."""
    return "--h:%.3f;" % max(0.0, min(1.0, pct / 100.0))


def md_table_to_html(block, index):
    """Render one pipe table. Tables 0, 1 and 2 of the comparison get their own styling."""
    head = _table_cells(block[0])
    aligns = []
    for c in _table_cells(block[1]):
        left, right = c.startswith(":"), c.endswith(":")
        aligns.append("center" if left and right else "right" if right else "left" if left else "")
    kind = {0: "ranking", 1: "cutoffs", 2: "tasks"}.get(index)

    def cell(tag, text, col, cls=None, extra=""):
        attrs = ""
        if cls:
            attrs += ' class="%s"' % cls
        style = ("text-align:%s;" % aligns[col] if col < len(aligns) and aligns[col] else "") + extra
        if style:
            attrs += ' style="%s"' % style
        return "<%s%s>%s</%s>" % (tag, attrs, md_inline(text), tag)

    out = ['<div class="table-wrap">', '<table%s>' % (' class="%s"' % kind if kind else ""),
           "<thead>", "<tr>" + "".join(cell("th", c, i) for i, c in enumerate(head)) + "</tr>",
           "</thead>", "<tbody>"]
    for line in block[2:]:
        cells = _table_cells(line)
        tr_cls = ""
        parts = []
        for i, text in enumerate(cells):
            cls, extra = None, ""
            if kind == "ranking" and len(cells) >= 12:
                cls = "rank" if i == 0 else "model" if i == 1 else None
            elif kind == "cutoffs":
                cls = "rank" if i == 0 else "model" if i == 1 else None
            elif kind == "tasks":
                cls = "task" if i == 0 else None
            heated = (kind == "ranking" and len(cells) >= 12 and 2 <= i < 12) or (kind == "tasks" and i > 0)
            m = HTML_NUM.match(text) if heated else None
            if m:
                cls = "heat"
                extra = _heat(float(m.group(1)))
            parts.append(cell("td", text, i, cls, extra))
        if kind == "ranking" and len(cells) >= 12 and cells[0] in ("1", "2", "3"):
            tr_cls = ' class="podium p%s"' % cells[0]
        out.append("<tr%s>%s</tr>" % (tr_cls, "".join(parts)))
    out += ["</tbody>", "</table>", "</div>"]
    return "\n".join(out)


def md_quote_to_html(lines):
    """Render a blockquote; a first line [!NOTE] (and the like) makes it a callout."""
    paras, cur = [], []
    for line in lines + [""]:
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paras.append(cur)
            cur = []
    kind = None
    if paras:
        m = CALLOUT_MARKER.match(paras[0][0])
        if m:
            kind = m.group(1).upper()
            paras[0] = paras[0][1:]
            if not paras[0]:
                paras.pop(0)
    body = "".join("<p>%s</p>" % md_inline(" ".join(p)) for p in paras)
    if kind is None:
        return "<blockquote>%s</blockquote>" % body
    label, path = CALLOUTS[kind]
    title = ('<p class="callout-title"><svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">'
             '<path fill="currentColor" d="%s"/></svg>%s</p>' % (path, label))
    return '<blockquote class="callout %s">%s%s</blockquote>' % (kind.lower(), title, body)


def md_to_html(md):
    """Convert comparison.md (headings, paragraphs, pipe tables, blockquotes) to an HTML fragment."""
    lines = md.splitlines()
    out, i, tables = [], 0, 0

    def starts_block(s):
        return s.startswith(("|", ">", "#"))

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(md_table_to_html(lines[i:j], tables))
            tables += 1
            i = j
        elif line.startswith(">"):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            out.append(md_quote_to_html([re.sub(r"^>\s?", "", s) for s in lines[i:j]]))
            i = j
        elif line.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            level = len(m.group(1)) if m else 1
            out.append("<h%d>%s</h%d>" % (level, md_inline(m.group(2) if m else line), level))
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not starts_block(lines[j]):
                j += 1
            out.append("<p>%s</p>" % md_inline(" ".join(s.strip() for s in lines[i:j])))
            i = j
    return "\n".join(out)


def comparison_html(md):
    """Wrap comparison.md in a self-contained, styled HTML page (heat-map cells, callouts, favicon)."""
    m = re.search(r"^# (.+)$", md, re.M)
    title = html_escape(m.group(1) if m else "3L2Mark Report", quote=False)
    icon = "data:image/png;base64," + FAVICON_PNG_B64
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="author" content="3L2Mark">
<meta name="copyright" content="&copy; 2026 Daniel Dias Rodrigues">
<link rel="icon" type="image/png" href="%s">
<title>%s</title>
<style>%s</style>
</head>
<body>
<main class="page">
%s
</main>
</body>
</html>
""" % (icon, title, HTML_CSS, md_to_html(md))


def compare(out_dir, show_all=False):
    """Build a ranking from every results.json found under out_dir."""
    found = []
    for root, _, files in os.walk(out_dir):
        if "results.json" in files:
            try:
                with open(os.path.join(root, "results.json"), encoding="utf-8") as f:
                    found.append(json.load(f))
            except Exception:
                pass
    if not found:
        print("No results found under %s" % out_dir)
        return 1
    # A migrated run keeps the date of its source: the higher version is the later one.
    found.sort(key=lambda d: (d["meta"]["date"], beta_number(d["meta"].get("version")) or 0))
    if not show_all:
        latest = {}
        for d in found:
            latest[d["meta"]["name"]] = d
        found = list(latest.values())
    by_id = {t["id"]: t for t in TASKS}
    rows, extras = [], []
    for d in found:
        tasks = [by_id[r["task"]] for r in d["results"] if r["task"] in by_id]
        seen, ordered = set(), []
        for t in TASKS:
            if t in tasks and t["id"] not in seen:
                seen.add(t["id"])
                ordered.append(t)
        rows.append((d["meta"], aggregate(d["results"], ordered)))
        extras.append(cutoff_stats(d["results"]))
    order = sorted(range(len(rows)), key=lambda i: -rows[i][1]["final"])
    rows = [rows[i] for i in order]
    extras = [extras[i] for i in order]
    cats = ["write", "bugfix", "optimize", "sql"]
    # SQL is both a category and a language with the same tasks: show it once (category).
    langs = ["Python", "C/C++", "C#", "PHP", "Java"]
    L = ["# 3L2Mark Report — Model comparison results", "",
         "Generated %s from %d result set(s)%s." % (
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), len(rows),
             "" if show_all else " (latest run of each model)"), "",
         "| # | Model | Score | Write | Bugs | Optimize | SQL | " + " | ".join(langs) +
         " | Solved | Builds | tok/s | Gen time | Runs | Date | Version |",
         "|---:|---|---:|---:|---:|---:|---:|" + "---:|" * len(langs) + "---:|---:|---:|---:|---:|---|---|"]
    for i, (meta, agg) in enumerate(rows, 1):
        cat = {k: None for k in cats}
        for label, sc, _ in agg["by_category"]:
            for k in cats:
                if CATEGORY_LABEL[k] == label:
                    cat[k] = sc
        lang = dict((label, sc) for label, sc, _ in agg["by_language"])
        score = "%.1f" % agg["final"]
        if agg["stdev"] is not None:
            score += " ±%.1f" % agg["stdev"]
        flags = ""
        if meta.get("partial") or meta.get("skipped_langs"):
            flags = " (incomplete)"
        L.append("| %d | %s%s | %s | %s | %s | %d/%d | %.0f%% | %s | %s | %d | %s | %s |" % (
            i, meta["name"], flags, score,
            " | ".join("-" if cat[k] is None else "%.1f" % cat[k] for k in cats),
            " | ".join("-" if l not in lang else "%.1f" % lang[l] for l in langs),
            agg["perfect"], len(agg["tasks"]), agg["compile_rate"],
            fmt_tps(agg["tps"], agg["tps_stdev"]),
            fmt_secs(agg["gen_seconds"]), meta["runs"], meta["date"][:10],
            meta.get("version", "unknown")))
    total_gen = sum(agg["gen_seconds"] for _, agg in rows)
    L += ["", "Total test time (sum of all Gen time): %s (%.1f hours)." % (fmt_secs(total_gen), total_gen / 3600.0)]
    L += ["", "> [!note]",
          "> **Solved**: how many tasks the model got completely right in every run (the model always gets that task right, in each run).",
          ">",
          "> **Builds**: percentage of all answers (tasks x runs) that compiled, or were accepted by the "
          "interpreter, on the first attempt, before any test ran. It does not say the code is correct: "
          "a high Builds with a low Score means the code builds but its logic is wrong."]
    versions = sorted(set(meta.get("version", "unknown") for meta, _ in rows))
    if len(set(version_group(v) for v in versions)) > 1:
        L += ["", "**Warning: these results come from different benchmark versions (%s). Tasks, tests "
                  "or limits may differ between versions, so the scores are not directly comparable.**"
                  % ", ".join(versions)]
    L += ["", "## Cut-offs and finished answers", "",
          "| # | Model | Score | Cut off (%s) | Score of `stop` answers only " % " + ".join("`%s`" % k for k in CUT_OFF) +
          "| Median tokens of `stop` answers |",
          "|:---:|---|---:|---|---:|---:|"]
    for i, ((meta, agg), ex) in enumerate(zip(rows, extras), 1):
        score = "%.1f" % agg["final"]
        if agg["stdev"] is not None:
            score += " ±%.1f" % agg["stdev"]
        kinds = ", ".join("%d `%s`" % (n, k) for k, n in ex["kinds"].items() if n)
        cut = "%d of %d%s" % (ex["cut"], ex["total"], " (%s)" % kinds if kinds else "")
        L.append("| %d | %s | %s | %s | %s | %s |" % (
            i, meta["name"], score, cut,
            "-" if ex["stop_score"] is None else "%.1f (n=%d)" % (ex["stop_score"], ex["stop_n"]),
            "-" if ex["stop_tokens"] is None else "%.0f" % ex["stop_tokens"]))
    L += ["", "> [!note]",
          "> (n=) is the number of `stop` answers (finished, not cut off) used for that score."]
    # Columns are headed by the rank number of the main table: model names are too wide.
    L += ["", "## Score per task", "",
          "Columns are the rank numbers (#) of the main table.", "",
          "| Task | " + " | ".join("%d" % i for i in range(1, len(rows) + 1)) + " |",
          "|---|" + "---:|" * len(rows)]
    for t in TASKS:
        cells = []
        for _, agg in rows:
            hit = [r for r in agg["tasks"] if r["id"] == t["id"]]
            cells.append("%.0f" % (100 * hit[0]["score"]) if hit else "-")
        L.append("| %s | %s |" % (t["id"], " | ".join(cells)))
    L += ["", "> [!note]",
          "> With one run per task, differences of a few points between two models are within "
          "sampling noise; use `--runs 5` (or more) and compare the mean and the standard deviation.", ""]
    text = "\n".join(L)
    path = os.path.join(out_dir, "comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    html_path = os.path.join(out_dir, "comparison.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(comparison_html(text))
    print(text)
    print("Saved to %s and %s" % (path, html_path))
    return 0


# ==========================================================================
# Main
# ==========================================================================

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Objective coding benchmark for a model served by llama.cpp (OpenAI-compatible API).")
    p.add_argument("--url", default=DEFAULT_URL, help="server base URL (default: %(default)s)")
    p.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"), help="API key, if the server needs one")
    p.add_argument("--name", help="model name used in reports and folder names (default: taken from the server)")
    p.add_argument("--runs", type=int, default=1, help="answers requested per task; the score is the mean (default: 1)")
    p.add_argument("--seed", type=int, default=1234, help="base seed sent to the server (run k uses seed+k)")
    p.add_argument("--max-tokens", type=int, default=24576, help="token cap per answer, reasoning included (default: %(default)s)")
    p.add_argument("--no-loop-detection", action="store_true",
                   help="do not stop an answer that repeats the same text over and over "
                        "(by default it is stopped early and reported with finish reason 'loop')")
    p.add_argument("--task-timeout", type=int, default=1800, help="wall-clock cap per answer in seconds (default: %(default)s)")
    p.add_argument("--temperature", type=float, help="override the server's temperature (default: do not send)")
    p.add_argument("--top-p", type=float, help="override the server's top_p (default: do not send)")
    p.add_argument("--extra-json", help="JSON object merged into every request, e.g. '{\"reasoning_effort\":\"high\"}'")
    p.add_argument("--system", help="optional system prompt")
    p.add_argument("--only", help="comma-separated task ids or substrings (e.g. sql,java_)")
    p.add_argument("--list", action="store_true", help="list the tasks and exit")
    p.add_argument("--out", default="results", help="results folder (default: ./results)")
    p.add_argument("--time-scale", type=float, default=1.0, help="multiply every time limit (use >1 on slow machines)")
    p.add_argument("--csharp", choices=("auto", "mcs", "dotnet"), default="auto", help="C# toolchain to use")
    p.add_argument("--skip-missing", action="store_true", help="skip languages whose toolchain is missing instead of aborting")
    p.add_argument("--keep-work", action="store_true", help="keep the build folder of every answer inside the results folder")
    p.add_argument("--resume", metavar="DIR", help="continue an interrupted run stored in DIR")
    p.add_argument("--records", metavar="DIR",
                   help="list the stored records of the run in DIR, one line each, and exit")
    p.add_argument("--remove", type=int, metavar="N",
                   help="with --records: delete record number N (as listed) so that --resume asks it again")
    p.add_argument("--migrate", metavar="DIR",
                   help="copy the run in DIR, made by an older compatible version, to DIR-betaN "
                        "without the records of the tasks that changed since, and exit; then "
                        "--resume the copy to ask only those again")
    p.add_argument("--selftest", action="store_true", help="validate toolchains and tasks with the reference solutions (no model)")
    p.add_argument("--compare", action="store_true", help="rank all models found in the results folder and exit")
    p.add_argument("--compare-all", action="store_true", help="like --compare but lists every stored run")
    return p.parse_args(argv)


def records_cmd(run_dir, remove):
    """List the records of a stored run, or delete one so that --resume asks it again."""
    path = os.path.join(run_dir.rstrip("/"), "results.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Cannot read %s: %s" % (path, e))
        return 2
    recs = data["results"]
    if remove is not None:
        if not 1 <= remove <= len(recs):
            print("No record number %d (the run has %d)." % (remove, len(recs)))
            return 2
        gone = recs.pop(remove - 1)
        shutil.copyfile(path, path + ".bak")
        data["meta"]["partial"] = True
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        print("Removed record %d: %s run %d (backup: %s.bak)." % (remove, gone["task"], gone["run"], path))
        print("Ask it again with:  python3 %s --resume %s --runs %s  (same options as the original run)"
              % (sys.argv[0], run_dir.rstrip("/"), data["meta"].get("runs", 1)))
        return 0
    m = data["meta"]
    print("%s   %s   %s   %d record(s)\n" % (m.get("name"), m.get("date"), m.get("version"), len(recs)))
    for i, r in enumerate(recs, 1):
        g = r.get("gen") or {}
        note = "  REQUEST FAILED" if request_failed(r) else ""
        print("%3d  %-26s run %-2d %6.1f%%  %2d/%-2d tests  gen %s, %s tok, %.1f tok/s  %s%s" % (
            i, r["task"], r["run"], 100 * r["score"], r["passed"], r["total"],
            fmt_secs(g.get("seconds") or 0), g.get("completion_tokens", 0), g.get("tps") or 0,
            g.get("finish"), note))
    return 0


def migrate_cmd(run_dir):
    """Copy a run of an older compatible version into a new folder, without the records of
    the tasks that changed since then, so that --resume asks only those again."""
    run_dir = run_dir.rstrip("/")
    path = os.path.join(run_dir, "results.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Cannot read %s: %s" % (path, e))
        return 2
    old_version = data["meta"].get("version", "unknown")
    if version_group(old_version) == version_group(VERSION):
        print("%s was made by codebench %s, which is compatible with %s: nothing to migrate."
              % (run_dir, old_version, VERSION))
        return 0
    if beta_number(old_version) not in MIGRATE_FROM:
        print("Cannot migrate %s: codebench %s can only migrate runs of Beta %s, and this one "
              "was made by %s." % (run_dir, VERSION, ", ".join(map(str, MIGRATE_FROM)), old_version))
        return 2
    dest = "%s-beta%d" % (run_dir, beta_number(VERSION))
    if os.path.exists(dest):
        print("Cannot migrate: %s already exists." % dest)
        return 2
    prefixes = tuple("%s_run" % tid for tid in MIGRATE_REDO)

    def ignore(folder, names):
        skip = {n for n in names if n.startswith(prefixes)}
        if os.path.abspath(folder) == os.path.abspath(run_dir):
            skip |= {"results.json", "results.json.bak", "report.md"} & set(names)
        return skip

    shutil.copytree(run_dir, dest, ignore=ignore)
    kept = [r for r in data["results"] if r["task"] not in MIGRATE_REDO]
    meta = dict(data["meta"])
    meta["version"] = VERSION
    meta["partial"] = True
    meta["migrated_from"] = {"version": old_version, "dir": run_dir, "redo": list(MIGRATE_REDO),
                             "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(dest, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": kept}, f, indent=1)
    write_report(os.path.join(dest, "report.md"), meta, kept, list(TASKS))
    print("Migrated %s (%s) to %s (%s)." % (run_dir, old_version, dest, VERSION))
    print("Kept %d record(s), dropped %d of the changed tasks (%s). The source folder is untouched."
          % (len(kept), len(data["results"]) - len(kept), ", ".join(MIGRATE_REDO)))
    print("Ask them again with:  python3 %s --resume %s --runs %s  (same options as the original run)"
          % (sys.argv[0], dest, meta.get("runs", 1)))
    return 0


def select_tasks(only):
    if not only:
        return list(TASKS)
    keys = [k.strip() for k in only.split(",") if k.strip()]
    return [t for t in TASKS if any(k == t["id"] or k in t["id"] for k in keys)]


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.compare or args.compare_all:
        return compare(args.out, args.compare_all)
    if args.remove is not None and not args.records:
        print("--remove needs --records DIR")
        return 2
    if args.records:
        return records_cmd(args.records, args.remove)
    if args.migrate:
        return migrate_cmd(args.migrate)

    tasks = select_tasks(args.only)
    if args.list:
        for t in tasks:
            print("%-22s %-7s %-9s %s" % (t["id"], LANG_LABEL[t["lang"]], t["category"], t["title"]))
        return 0
    if not tasks:
        print("No task matches --only %s" % args.only)
        return 2

    tc = detect_toolchains(args.csharp)
    needed = []
    for t in tasks:
        if t["lang"] not in needed:
            needed.append(t["lang"])
    missing = [l for l in needed if not tc[l]["ok"]]
    print("codebench %s" % VERSION)
    for l in needed:
        extra = ""
        if l in ("c", "cpp") and tc[l]["ok"] and not tc[l].get("san"):
            extra = "  (sanitizers unavailable: memory errors will not be detected)"
        print("  %-7s %s%s" % (LANG_LABEL[l], tc[l]["desc"] if tc[l]["ok"] else "MISSING", extra))
    if missing:
        hints = " ".join(sorted(set(tc[l]["hint"].split(" (")[0] for l in missing if l != "sql")))
        print("\nMissing toolchains: %s" % ", ".join(LANG_LABEL[l] for l in missing))
        if hints:
            print("On Debian/Ubuntu:  sudo apt install %s" % hints)
        if not args.skip_missing:
            print("Install them, or run again with --skip-missing (the score is then not comparable "
                  "with full runs).")
            return 2
        tasks = [t for t in tasks if tc[t["lang"]]["ok"]]

    if args.selftest:
        return selftest(tasks, tc, args.time_scale)

    # ---- talk to the server -------------------------------------------------
    server = Server(args.url, args.api_key)
    info = server.probe()
    if info is None:
        print("\nNo server answering at %s. Start the model first, e.g.:\n"
              "  llama serve -m model.gguf --port 9931 -c 32768 ...\n"
              "(any OpenAI-compatible endpoint works; change the address with --url)" % server.url)
        return 2
    extra = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.top_p is not None:
        extra["top_p"] = args.top_p
    if args.extra_json:
        try:
            extra.update(json.loads(args.extra_json))
        except ValueError as e:
            print("--extra-json is not valid JSON: %s" % e)
            return 2

    name = args.name or slugify(info.get("model_id") or info.get("model_path") or "model")
    results = []
    if args.resume:
        run_dir = args.resume.rstrip("/")
        try:
            with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
                old = json.load(f)
            # A request that failed says nothing about the model: ask again.
            results = [r for r in old["results"] if not request_failed(r)]
            retried = len(old["results"]) - len(results)
            name = old["meta"]["name"]
            date = old["meta"]["date"]
        except Exception as e:
            print("Cannot resume from %s: %s" % (run_dir, e))
            return 2
        old_version = old["meta"].get("version", "unknown")
        if version_group(old_version) != version_group(VERSION):
            print("Cannot resume from %s: it was created by codebench %s and this is %s; "
                  "scores from different versions must not be mixed in one run."
                  % (run_dir, old_version, VERSION))
            return 2
        if retried:
            print("Resuming: %d answer(s) lost to request errors will be asked again." % retried)
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_dir = os.path.join(args.out, slugify(name),
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(run_dir, "answers"), exist_ok=True)

    plan = [(t, k) for k in range(1, args.runs + 1) for t in tasks]
    done = set((r["task"], r["run"]) for r in results)
    meta = {
        "name": name, "date": date, "url": server.url, "server": info, "overrides": extra,
        "runs": args.runs, "seed": args.seed, "max_tokens": args.max_tokens,
        "task_timeout": args.task_timeout, "scale": args.time_scale, "machine": machine_info(),
        "toolchains": {l: tc[l]["desc"] + (" + ASan/UBSan" if tc[l].get("san") else "")
                       for l in needed if tc[l]["ok"]},
        "skipped_langs": [LANG_LABEL[l] for l in missing], "planned": len(plan), "version": VERSION,
    }
    if args.resume and old["meta"].get("migrated_from"):
        meta["migrated_from"] = old["meta"]["migrated_from"]

    def save(partial):
        meta["partial"] = partial
        slim = []
        for r in results:
            slim.append({k: v for k, v in r.items() if k != "code"})
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": slim}, f, indent=1)
        return write_report(os.path.join(run_dir, "report.md"), meta, results, tasks)

    print("\nModel: %s   (server: %s%s)" % (name, server.url,
                                            ", ctx %s" % info["n_ctx"] if info.get("n_ctx") else ""))
    print("Tasks: %d x %d run(s). Results in %s\n" % (len(tasks), args.runs, run_dir))
    ctx_needed = args.max_tokens + 4096  # answer cap plus room for the longest prompt
    if info.get("n_ctx") and int(info["n_ctx"]) < ctx_needed:
        print("WARNING: context size %s is small; long answers will be cut off (%d or more is "
              "recommended for --max-tokens %d).\n" % (info["n_ctx"], ctx_needed, args.max_tokens))

    consecutive_errors = 0
    t_start = time.monotonic()
    try:
        for idx, (t, k) in enumerate(plan, 1):
            if (t["id"], k) in done:
                continue
            label = "[%2d/%d] %-22s" % (idx, len(plan), t["id"]) + (" run %d" % k if args.runs > 1 else "")
            prompt = build_prompt(t, tc)
            messages = ([{"role": "system", "content": args.system}] if args.system else []) + \
                       [{"role": "user", "content": prompt}]

            def progress(chunks, secs, label=label):
                status("%s generating... %d tokens, %s" % (label, chunks, fmt_secs(secs)))

            gen = server.chat(messages, args.max_tokens, args.seed + k, args.task_timeout, extra,
                              progress=progress, detect_loops=not args.no_loop_detection)
            status("%s testing..." % label)
            stem = os.path.join(run_dir, "answers", "%s_run%d" % (t["id"], k))
            with open(stem + ".md", "w", encoding="utf-8") as f:
                f.write("# %s (run %d)\n\n## Prompt\n\n%s\n\n" % (t["id"], k, prompt))
                if gen["reasoning"]:
                    f.write("## Reasoning\n\n%s\n\n" % gen["reasoning"])
                f.write("## Answer\n\n%s\n" % gen["content"])
            # A fragment left by a failed request is not an answer: it scores 0 until asked again.
            res = grade_answer(t, "" if gen["error"] else gen["content"], tc, args.time_scale,
                               keep_dir=(stem + "_build") if args.keep_work and t["lang"] != "sql" else None)
            ext = {"python": "py", "c": "c", "cpp": "cpp", "csharp": "cs", "php": "php", "java": "java",
                   "sql": "sql"}[t["lang"]]
            with open(stem + "." + ext, "w", encoding="utf-8") as f:
                f.write(res["code"])
            rec = {"task": t["id"], "run": k, "score": res["score"], "passed": res["passed"],
                   "total": res["total"], "compiled": res["compiled"], "compile_log": res["compile_log"],
                   "warnings": res["warnings"], "extraction": res["extraction"], "tests": res["tests"],
                   "gen": {kk: gen[kk] for kk in ("seconds", "completion_tokens", "prompt_tokens", "tps",
                                                  "finish", "error")}}
            results.append(rec)
            clear_status()
            note = ""
            if gen["error"]:
                note = "  REQUEST ERROR: %s" % shorten(gen["error"], 120)
            elif gen["finish"] in CUT_OFF:
                note = "  (answer cut off: %s)" % gen["finish"]
            print("%s %5.1f%%  %2d/%-2d tests  %s  gen %s, %d tok, %.1f tok/s%s" % (
                label, 100 * res["score"], res["passed"], res["total"],
                "build ok  " if res["compiled"] else "NO BUILD  ", fmt_secs(gen["seconds"]),
                gen["completion_tokens"] or 0, gen["tps"] or 0.0, note))
            sys.stdout.flush()
            save(partial=True)
            consecutive_errors = consecutive_errors + 1 if gen["error"] else 0
            if consecutive_errors >= 3:
                print("\nThree requests failed in a row; stopping. Fix the server and continue with:\n"
                      "  python3 %s --resume %s" % (sys.argv[0], run_dir))
                return 1
    except KeyboardInterrupt:
        clear_status()
        if results:
            save(partial=True)
        print("\nInterrupted. Continue later with:  python3 %s --resume %s" % (sys.argv[0], run_dir))
        return 130

    agg = save(partial=len(results) < len(plan))
    lost = sum(1 for r in results if request_failed(r))
    if lost:
        print("\nWARNING: %d request(s) failed and count as 0. Ask them again with:\n"
              "  python3 %s --resume %s" % (lost, sys.argv[0], run_dir))
    print("\n" + "=" * 64)
    print("FINAL SCORE: %.1f / 100   (%s)" % (agg["final"], name))
    if agg["stdev"] is not None:
        print("  per run: %s   standard deviation %.1f" % (", ".join("%.1f" % x for x in agg["per_run"]),
                                                           agg["stdev"]))
    for label, sc, n in agg["by_category"]:
        print("  %-34s %5.1f" % (label, sc))
    print("  %d of %d tasks solved completely; total time %s" % (
        agg["perfect"], len(agg["tasks"]), fmt_secs(time.monotonic() - t_start)))
    print("Report: %s" % os.path.join(run_dir, "report.md"))
    print("Compare models with:  python3 %s --compare" % sys.argv[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
