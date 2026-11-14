"""Terminal dashboard for training runs (start it with: tools/ygo.sh tui).

Shows the state of a run folder under ~/ygo/runs (steps, SPS, returns, eval win rate, checkpoints,
GPU, background jobs, log tail) and controls it through tools/ygo.sh.

Keys:  s resume a stopped run     N start a new run     x stop (clean, saves a checkpoint)
       r record a match           e eval vs the promoted model (quick subset)
       t TensorBoard              n next run            q quit (training keeps running)

usage: python tools/ygo_tui.py [RUN] [--once]     (--once prints one frame and exits)
"""
import os, re, select, subprocess, sys, time
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REPO = Path(__file__).resolve().parents[1]
YGO = Path(os.environ.get("YGO_HOME", Path.home() / "ygo"))
RUNS = YGO / "runs"
JOBS = RUNS / ".jobs"
YGO_SH = str(REPO / "tools" / "ygo.sh")
BARS = "▁▂▃▄▅▆▇█"
STATE_STYLE = {"running": "bold green", "stopping": "bold yellow", "stopped": "bold cyan",
               "crashed": "bold red", "new": "dim"}


def sh(*args, timeout=20):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001 - the dashboard must keep running
        return f"error: {e}"


def tmux_has(name):
    return subprocess.run(["tmux", "has-session", "-t", "=" + name], capture_output=True).returncode == 0


def run_names():
    return sorted((d.name for d in RUNS.iterdir() if d.is_dir() and not d.name.startswith(".")),
                  key=lambda n: (RUNS / n).stat().st_mtime, reverse=True) if RUNS.is_dir() else []


def state(run):
    d = RUNS / run
    if tmux_has(run):
        return "stopping" if (d / "checkpoints" / "STOP").exists() else "running"
    wl = d / "watchdog.log"
    if not wl.exists() or wl.stat().st_size == 0:
        # runs from before the watchdog have a train.log but no watchdog.log
        return "stopped" if (d / "train.log").exists() else "new"
    return "stopped" if "exited normally" in wl.read_text(errors="replace").strip().splitlines()[-1] else "crashed"


def tail_text(path, nbytes=400_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def spark(values, width=40):
    values = [v for v in values if v == v][-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return "".join(BARS[int((v - lo) / (hi - lo) * 7)] if hi > lo else BARS[3] for v in values)


def nums(pattern, text, cast=float):
    return [cast(x) for x in re.findall(pattern, text)]


def fmt_age(seconds):
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{seconds // 60} min"
    return f"{seconds / 3600:.1f} h"


def collect(run):
    d = RUNS / run
    log = tail_text(d / "train.log")
    steps = nums(r"global_step=(\d+)", log, int)
    cks = sorted((d / "checkpoints").glob("*.flax_model"), key=lambda p: p.stat().st_mtime) if (d / "checkpoints").is_dir() else []
    info = {
        "state": state(run), "step": steps[-1] if steps else None, "steps": steps,
        "sps": nums(r"SPS: (\d+)", log), "ret": nums(r"avg_return=(-?[0-9.]+)", log),
        "length": nums(r"avg_length=(\d+)", log), "eval": nums(r"eval_win_rate=([0-9.]+)", log),
        "nonfinite": log.count("WARNING: non-finite"), "cks": cks,
        "log_age": time.time() - (d / "train.log").stat().st_mtime if (d / "train.log").exists() else None,
        "watchdog": tail_text(d / "watchdog.log", 2000).strip().splitlines()[-2:],
        "loglines": [l for l in log.splitlines() if l.strip() and not l.startswith(("W0000", "I0000"))][-7:],
    }
    return info


def gpu_line():
    out = sh("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits", timeout=5)
    try:
        u, used, total, temp = [x.strip() for x in out.split(",")]
        return f"{u}% util, {int(used) / 1024:.1f} of {int(total) / 1024:.1f} GB, {temp} °C"
    except ValueError:
        return "n/a"


def job_lines():
    rows = []
    for name in ("record", "eval"):
        log = JOBS / f"{name}.log"
        running = tmux_has("ygo_" + name)
        if running or log.exists():
            last = (tail_text(log, 3000).strip().splitlines() or [""])[-1]
            rows.append((name, "running" if running else "done", last[:100]))
    if tmux_has("ygo_tb"):
        rows.append(("tensorboard", "running", "http://localhost:6006"))
    return rows


def render(run, info, message):
    st = info["state"]
    head = Text.assemble(("  ygo-agent training   ", "bold"), ("run ", "dim"), (run, "bold"), "   ",
                         (st.upper(), STATE_STYLE.get(st, "")), "   ", (time.strftime("%H:%M:%S"), "dim"))
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", no_wrap=True); t.add_column()
    step = info["step"]
    t.add_row("steps", f"{step:,} ({step / 2**20:.1f}M)" if step else "-")
    sps = info["sps"]
    t.add_row("SPS", f"{int(sps[-1]):,}  {spark(sps)}" if sps else "-")
    t.add_row("avg return", f"{info['ret'][-1]:+.3f}  {spark(info['ret'])}" if info["ret"] else "-")
    t.add_row("avg game length", f"{int(info['length'][-1])} steps" if info["length"] else "-")
    ev = info["eval"]
    t.add_row("eval vs built-in bot", f"{ev[-1]:.3f} (mean of last 10: {sum(ev[-10:]) / len(ev[-10:]):.3f})  {spark(ev)}" if ev else "-")
    nf = info["nonfinite"]
    t.add_row("dropped updates", Text(f"{nf} iterations with non-finite losses", style="red" if nf else ""))
    if info["log_age"] is not None:
        stale = st == "running" and info["log_age"] > 300
        t.add_row("last log line", Text(fmt_age(info["log_age"]) + " ago" + (" (no output for over 5 minutes)" if stale else ""),
                                        style="red" if stale else ""))
    c = Table.grid(padding=(0, 2))
    c.add_column(style="dim", no_wrap=True); c.add_column()
    cks = info["cks"]
    if cks:
        size = sum(p.stat().st_size for p in cks) / 2**30
        c.add_row("newest checkpoint", f"{cks[-1].name}  ({fmt_age(time.time() - cks[-1].stat().st_mtime)} ago)")
        c.add_row("checkpoints", f"{len(cks)} files, {size:.1f} GB")
    else:
        c.add_row("checkpoints", "none yet")
    c.add_row("GPU", gpu_line())
    for line in info["watchdog"]:
        c.add_row("watchdog", line[:110])
    jobs = job_lines()
    j = Table.grid(padding=(0, 2))
    for name, s, last in jobs:
        j.add_row(Text(name, style="bold"), Text(s, style="green" if s == "running" else "dim"), last)
    keys = Text.from_markup("[bold]s[/] resume   [bold]N[/] new run   [bold]x[/] stop   [bold]r[/] record   "
                            "[bold]e[/] eval   [bold]t[/] TensorBoard   [bold]n[/] next run   [bold]q[/] quit")
    parts = [head, Panel(t, title="training", border_style="green" if st == "running" else "dim"),
             Panel(c, title="checkpoints and system")]
    if jobs:
        parts.append(Panel(j, title="jobs"))
    parts.append(Panel(Text("\n".join(l[:150] for l in info["loglines"]) or "(no log yet)", style="dim"), title="train.log"))
    if message:
        parts.append(Panel(Text.from_markup(message), border_style="yellow"))
    parts.append(keys)
    return Group(*parts)


def start_job(name, *cmd):
    if tmux_has("ygo_" + name):
        return f"a {name} job is already running"
    JOBS.mkdir(parents=True, exist_ok=True)
    log = JOBS / f"{name}.log"
    quoted = " ".join("'" + c.replace("'", "'\\''") + "'" for c in cmd)
    subprocess.run(["tmux", "new-session", "-d", "-s", "ygo_" + name, f"{quoted} > '{log}' 2>&1"])
    return f"{name} started in the background (log: {log})"


def read_key(timeout):
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    once = "--once" in sys.argv
    names = run_names()
    cur_file = RUNS / ".current"
    run = args[0] if args else (cur_file.read_text().strip() if cur_file.exists() else (names[0] if names else None))
    console = Console()
    if run is None or not (RUNS / run).is_dir():
        if once or not sys.stdin.isatty():
            console.print("no run folder yet; start one with: tools/ygo.sh start NAME"); return
        run = None
    if once:
        console.print(render(run, collect(run), "")); return
    if not sys.stdin.isatty():
        console.print("the dashboard needs a terminal; use --once for a single frame"); return
    import termios, tty
    old = termios.tcgetattr(sys.stdin)
    message, pending, decks = "", None, []
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(console=console, screen=True, auto_refresh=False) as live:
            last = 0.0
            while True:
                if time.time() - last > 2:
                    if run is None:
                        live.update(Panel(Text.from_markup("no run folder yet. Press [bold]N[/] to start one, [bold]q[/] to quit.")), refresh=True)
                    else:
                        live.update(render(run, collect(run), message), refresh=True)
                    last = time.time()
                k = read_key(0.2)
                if k is None:
                    continue
                last = 0.0
                if pending == "record":
                    pending = None
                    i = ord(k) - ord("a")
                    message = start_job("record", YGO_SH, "record", run, decks[i], "3") if 0 <= i < len(decks) else "record cancelled"
                elif pending in ("stop", "new"):
                    what, pending = pending, None
                    if k != "y":
                        message = "cancelled"
                    elif what == "stop":
                        message = sh(YGO_SH, "stop", run)
                    else:
                        out = sh(YGO_SH, "start", timeout=60)
                        names = run_names(); run = cur_file.read_text().strip() if cur_file.exists() else run
                        message = (out.splitlines() or ["started"])[0]
                elif k == "q":
                    break
                elif k == "N":
                    pending, message = "new", "Start a [bold]new[/] run from the promoted model with the default settings? Press [bold]y[/] to confirm."
                elif run is None:
                    continue
                elif k == "n":
                    names = run_names(); run = names[(names.index(run) + 1) % len(names)] if run in names else names[0]; message = ""
                elif k == "s":
                    st = state(run)
                    message = sh(YGO_SH, "resume", run) if st in ("stopped", "crashed") else f"{run} is {st}; only a stopped or crashed run can be resumed"
                elif k == "x":
                    if state(run) == "running":
                        pending, message = "stop", f"Stop [bold]{run}[/]? It finishes the iteration and saves a checkpoint. Press [bold]y[/] to confirm."
                    else:
                        message = f"{run} is not running"
                elif k == "r":
                    decks = sorted(p.stem for p in (YGO / "decks" / "mine").glob("*.ydk"))[:26]
                    pending = "record"
                    message = "Record 3 games, newest checkpoint vs the promoted model. Deck:\n" + "\n".join(
                        f"  [bold]{chr(ord('a') + i)}[/] {d}" for i, d in enumerate(decks)) + "\n(any other key cancels)"
                elif k == "e":
                    message = start_job("eval", YGO_SH, "eval", run)
                elif k == "t":
                    message = sh(YGO_SH, "tb") or "TensorBoard: http://localhost:6006"
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
