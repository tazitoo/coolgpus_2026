# Coolgpus Modernization for Ubuntu 24.04

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize coolgpus to work reliably on Ubuntu 24.04 with 3 multi-fan NVIDIA GPUs, fix known bugs, and improve usability.

**Architecture:** The script stays as a single-file CLI tool. We refactor it into a proper Python package with a `__main__.py` entry point, add proper fan-to-GPU discovery, make hardcoded values configurable, and ship a systemd service file. Tests use pytest with subprocess mocking.

**Tech Stack:** Python 3.10+, pytest, setuptools (pyproject.toml)

---

## Chunk 1: Project Structure and Packaging

### Task 1: Convert to pyproject.toml and package structure

**Files:**
- Create: `pyproject.toml`
- Create: `src/coolgpus/__init__.py`
- Create: `src/coolgpus/__main__.py`
- Create: `src/coolgpus/cli.py` (argument parsing)
- Create: `src/coolgpus/core.py` (fan control logic — curve calculation, clamping)
- Create: `src/coolgpus/nvidia.py` (all nvidia-smi/nvidia-settings subprocess calls)
- Create: `src/coolgpus/xserver.py` (X server management)
- Delete: `setup.py` (after migration)
- Delete: `coolgpus` (the old single-file script, after migration)

The old monolithic `coolgpus` script (316 lines) is split into focused modules. This makes it testable — the pure logic in `core.py` can be unit tested without mocking subprocesses.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "coolgpus"
version = "1.0.0"
description = "GPU fan control for headless Linux"
readme = "readme.md"
license = "MIT"
requires-python = ">=3.10"

[project.scripts]
coolgpus = "coolgpus.cli:main"

[project.optional-dependencies]
dev = ["pytest>=7.0"]
```

- [ ] **Step 2: Create `src/coolgpus/__init__.py`**

Empty file, just makes it a package.

- [ ] **Step 3: Create `src/coolgpus/__main__.py`**

```python
from coolgpus.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Extract core logic into `src/coolgpus/core.py`**

Move these pure functions from the old `coolgpus` script (no subprocess calls):
- `determine_segment()` (line 167-187)
- `min_speed()` (line 189-192)
- `max_speed()` (line 194-195)
- `target_speed()` (line 197-199)
- `clamp()` (line 231-232)

These functions currently depend on `args` and `T_HYST` as globals. Refactor them to accept parameters explicitly:

```python
def determine_segment(t, temps, speeds):
    """Determine which segment of the piecewise linear fan curve t belongs to."""
    segments = zip(temps[:-1], temps[1:], speeds[:-1], speeds[1:])
    for temp_a, temp_b, speed_a, speed_b in segments:
        if t < temp_a:
            break
        if temp_a <= t < temp_b:
            break
    return temp_a, temp_b, speed_a, speed_b


def min_speed(t, temps, speeds):
    temp_a, temp_b, speed_a, speed_b = determine_segment(t, temps, speeds)
    load = (t - temp_a) / float(temp_b - temp_a)
    return int(min(max(speed_a + (speed_b - speed_a) * load, speed_a), speed_b))


def max_speed(t, temps, speeds, hyst):
    return min_speed(t + hyst, temps, speeds)


def target_speed(current_speed, temp, temps, speeds, hyst):
    lo = min_speed(temp, temps, speeds)
    hi = max_speed(temp, temps, speeds, hyst)
    return min(max(current_speed, lo), hi), lo, hi


def clamp(v, lo, hi):
    return min(hi, max(lo, v))
```

- [ ] **Step 5: Extract nvidia commands into `src/coolgpus/nvidia.py`**

Move all subprocess-based nvidia interactions. Key change: add `discover_fan_to_gpu_map()` that queries nvidia-settings to build the correct fan-index-to-GPU-index mapping. Also accept `verbose` and `display` as parameters instead of globals.

```python
import re
from subprocess import TimeoutExpired, check_output, Popen, PIPE, STDOUT, DEVNULL


def log_output(command, verbose=False, ok=(0,), stderr=STDOUT, timeout=60):
    """Run a command, capture output, raise on failure."""
    output = []
    if verbose:
        print("Command launched: " + " ".join(command))
    p = Popen(command, stdout=PIPE, stderr=stderr)
    try:
        p.wait(timeout)
        for line in p.stdout:
            output.append(line.decode().strip())
            if verbose:
                print(line.decode().strip())
        if verbose:
            print("Command finished")
    except TimeoutExpired:
        print("Command timed out: " + " ".join(command))
        raise
    finally:
        if p.returncode not in ok:
            print("\n".join(output))
            raise ValueError(
                "Command crashed with return code "
                + str(p.returncode)
                + ": "
                + " ".join(command)
            )
        return "\n".join(output)


def gpu_buses(verbose=False):
    return log_output(
        ["nvidia-smi", "--format=csv,noheader", "--query-gpu=pci.bus_id"],
        verbose=verbose,
    ).splitlines()


def temperature(bus, verbose=False):
    [line] = log_output(
        ["nvidia-smi", "--format=csv,noheader", "--query-gpu=temperature.gpu", "-i", bus],
        verbose=verbose,
    ).splitlines()
    return int(line)


def discover_fan_to_gpu_map(display, verbose=False):
    """Query nvidia-settings to discover which fans belong to which GPU.

    Returns a dict mapping gpu_index -> list of fan_indices.
    This handles multi-fan GPUs correctly (e.g., GPU 0 has fans [0,1]).
    """
    # First, find how many fans exist
    output = log_output(
        ["nvidia-settings", "-q", "fans", "-c", display],
        verbose=verbose,
    )
    fan_indices = [int(m) for m in re.findall(r"\[fan:(\d+)\]", output)]

    gpu_to_fans = {}
    for fan_id in fan_indices:
        # Query which GPU this fan belongs to
        fan_output = log_output(
            ["nvidia-settings", "-q", f"[fan:{fan_id}]/GPUCurrentFanSpeed", "-c", display],
            verbose=verbose,
            stderr=DEVNULL,
        )
        gpu_match = re.search(r"\[gpu:(\d+)\]", fan_output)
        if gpu_match:
            gpu_id = int(gpu_match.group(1))
            gpu_to_fans.setdefault(gpu_id, []).append(fan_id)

    return gpu_to_fans


def get_fan_speed_ranges(fan_indices, display, verbose=False):
    """Get valid speed range for each fan index.

    Returns dict of fan_index -> (min_speed, max_speed).
    """
    ranges = {}
    for fan_id in fan_indices:
        output = log_output(
            ["nvidia-settings", "-q", f"[fan:{fan_id}]/GPUTargetFanSpeed", "-c", display],
            verbose=verbose,
        )
        match = re.search(r"are\sin\sthe\srange\s+(\d+)\s+-\s+(\d+)", output, re.DOTALL)
        if match:
            ranges[fan_id] = (int(match.group(1)), int(match.group(2)))
        else:
            raise ValueError(f"Couldn't get valid value for GPUTargetFanSpeed on fan:{fan_id}")
    return ranges


def set_fan_speed(gpu_id, fan_ids, target, display, verbose=False):
    """Enable manual fan control on a GPU and set all its fans to target speed."""
    log_output(
        ["nvidia-settings", "-a", f"[gpu:{gpu_id}]/GPUFanControlState=1", "-c", display],
        verbose=verbose,
    )
    for fan_id in fan_ids:
        log_output(
            ["nvidia-settings", "-a", f"[fan:{fan_id}]/GPUTargetFanSpeed={int(target)}", "-c", display],
            verbose=verbose,
        )


def release_fan_control(gpu_id, display, verbose=False):
    """Return fan control to the driver for a GPU."""
    log_output(
        ["nvidia-settings", "-a", f"[gpu:{gpu_id}]/GPUFanControlState=0", "-c", display],
        verbose=verbose,
    )


def fetch_current_fan_speed(fan_id, display, verbose=False):
    return int(
        log_output(
            ["nvidia-settings", "-q", f"[fan:{fan_id}]/GPUCurrentFanSpeed", "-c", display, "--terse"],
            verbose=verbose,
            stderr=DEVNULL,
        )
    )
```

- [ ] **Step 6: Extract X server management into `src/coolgpus/xserver.py`**

Move `config()`, `xserver()`, `xserver_pids()`, `kill_xservers()`, and the `xservers()` context manager. Same logic, but accept parameters instead of reading globals.

```python
import os
import time
from contextlib import contextmanager
from subprocess import DEVNULL, Popen
from tempfile import mkdtemp

from coolgpus.nvidia import log_output


def generate_xorg_config(verbose=False):
    """Generate X server config with cool-bits enabled for fan control."""
    tempdir = mkdtemp(prefix="cool-gpu")
    conf = os.path.join(tempdir, "xorg.conf")
    log_output(
        [
            "nvidia-xconfig",
            "--enable-all-gpus",
            "--cool-bits=4",
            "--allow-empty-initial-configuration",
            "-o",
            conf,
        ],
        verbose=verbose,
    )
    return conf


def start_xserver(display, verbose=False):
    """Start an X server on the given display."""
    conf = generate_xorg_config(verbose=verbose)
    xorgargs = ["Xorg", display, "-once", "-config", conf]
    print("Starting xserver: " + " ".join(xorgargs))
    p = Popen(xorgargs, stdout=DEVNULL, stderr=DEVNULL)
    if verbose:
        print("Started xserver")
    return p


def xserver_pids(verbose=False):
    return list(
        map(int, log_output(["pgrep", "Xorg"], ok=(0, 1), verbose=verbose).splitlines())
    )


def kill_xservers(kill=False, verbose=False):
    """Kill existing X servers if requested, or raise if they exist."""
    servers = xserver_pids(verbose=verbose)
    if servers:
        if kill:
            print("Killing all running X servers, including " + ", ".join(map(str, servers)))
            log_output(["pkill", "-9", "Xorg"], ok=(0, 1), verbose=verbose)
            for _ in range(10):
                if xserver_pids(verbose=verbose):
                    print("Awaiting X server shutdown")
                    time.sleep(1)
                else:
                    print("All X servers killed")
                    return
            raise IOError("Failed to kill existing X servers. Try killing them yourself before running this script")
        else:
            raise IOError(
                "There are already X servers active. Either run the script with the `--kill` switch, "
                "or kill them yourself first"
            )
    else:
        print("No existing X servers, we're good to go")


@contextmanager
def managed_xserver(display, kill=False, verbose=False):
    """Context manager that starts an X server and cleans up on exit."""
    kill_xservers(kill=kill, verbose=verbose)
    xserver_process = None
    try:
        xserver_process = start_xserver(display, verbose=verbose)
        yield None
    finally:
        if xserver_process:
            print(f"Terminating xserver for display {display}.")
            xserver_process.terminate()
```

- [ ] **Step 7: Create `src/coolgpus/cli.py`** with argument parsing and main loop

This is the entry point that wires everything together. Replaces `run()`, `manage_fans()`, and the global argparse setup.

```python
import argparse
import signal
import sys
import time

from coolgpus.core import clamp, target_speed
from coolgpus.nvidia import (
    discover_fan_to_gpu_map,
    get_fan_speed_ranges,
    gpu_buses,
    release_fan_control,
    set_fan_speed,
    temperature,
)
from coolgpus.xserver import managed_xserver


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="""\
GPU fan control for headless Linux.

By default, uses a clamped linear fan curve from 30% below 55C to 99% above 80C,
with a small hysteresis gap to reduce fan speed oscillation."""
    )
    parser.add_argument(
        "--temp", nargs="+", default=[55, 80], type=float,
        help="Temperature thresholds for piecewise linear fan curve",
    )
    parser.add_argument(
        "--speed", nargs="+", default=[30, 99], type=float,
        help="Fan speed percentages corresponding to each temperature threshold",
    )
    parser.add_argument(
        "--hyst", nargs="?", default=2, type=float,
        help="Hysteresis gap in celsius (default: 2)",
    )
    parser.add_argument(
        "--interval", type=float, default=7,
        help="Seconds between fan speed updates (default: 7)",
    )
    parser.add_argument("--kill", action="store_true", default=False, help="Kill existing Xorg sessions before starting")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print extra debugging information")
    parser.add_argument("--display", type=str, default=":0", help="X server display to use (default: :0)")
    parser.add_argument("--test", action="store_true", help="Run hardware test mode")
    args = parser.parse_args(argv)

    assert len(args.temp) == len(args.speed), "temp and speed should have the same length"
    assert len(args.temp) >= 2, "Please use at least two points for temp"
    assert len(args.speed) >= 2, "Please use at least two points for speed"

    return args


def manage_fans(args, buses, gpu_to_fans, fan_speed_ranges):
    """Main fan control loop. Adjusts fan speeds based on GPU temperatures."""
    # Track current speed per GPU
    speeds = {gpu_id: 0 for gpu_id in gpu_to_fans}

    try:
        while True:
            start_time = time.time()

            for gpu_id, fan_ids in gpu_to_fans.items():
                bus = buses[gpu_id]
                temp = temperature(bus, verbose=args.verbose)

                # Use the range from the first fan of this GPU (all fans on a GPU share limits)
                fan_min, fan_max = fan_speed_ranges[fan_ids[0]]

                s, lo, hi = target_speed(speeds[gpu_id], temp, args.temp, args.speed, args.hyst)
                s = clamp(s, fan_min, fan_max)
                lo = clamp(lo, fan_min, fan_max)
                hi = clamp(hi, fan_min, fan_max)

                if s != speeds[gpu_id]:
                    print(f"GPU {gpu_id}, {temp}C -> [{lo}%-{hi}%]. Setting speed to {s}%")
                    set_fan_speed(gpu_id, fan_ids, s, args.display, verbose=args.verbose)
                else:
                    print(f"GPU {gpu_id}, {temp}C -> [{lo}%-{hi}%]. Leaving speed at {s}%")

                speeds[gpu_id] = s

            elapsed = time.time() - start_time
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
    finally:
        for gpu_id in gpu_to_fans:
            release_fan_control(gpu_id, args.display, verbose=args.verbose)
            print(f"Released fan speed control for GPU {gpu_id}")


def test_mode(args, buses, gpu_to_fans, fan_speed_ranges):
    """Run hardware validation tests."""
    import random

    # Validate GPU count
    gpu_count = len(buses)
    assert len(gpu_to_fans) == gpu_count, f"GPU count mismatch: {len(gpu_to_fans)} vs {gpu_count}"

    # Validate fan speed ranges
    for fan_id, (lo, hi) in fan_speed_ranges.items():
        assert 0 <= lo <= 45, f"Fan {fan_id} min speed {lo} outside expected range 0-45"
        assert 75 <= hi <= 110, f"Fan {fan_id} max speed {hi} outside expected range 75-110"

    # Pick a random GPU and test fan control
    gpu_id = random.choice(list(gpu_to_fans.keys()))
    bus = buses[gpu_id]
    fan_ids = gpu_to_fans[gpu_id]
    temp = temperature(bus, verbose=args.verbose)
    assert 0 <= temp <= 100, f"GPU {gpu_id} temperature {temp}C outside expected range"

    fan_min, fan_max = fan_speed_ranges[fan_ids[0]]
    test_speed = clamp(90, fan_min, fan_max)

    print(f"Setting GPU {gpu_id} fans to {test_speed}%...")
    set_fan_speed(gpu_id, fan_ids, test_speed, args.display, verbose=args.verbose)
    time.sleep(20)

    from coolgpus.nvidia import fetch_current_fan_speed

    for fan_id in fan_ids:
        actual = fetch_current_fan_speed(fan_id, args.display, verbose=args.verbose)
        assert actual > test_speed - 10, f"Fan {fan_id} speed {actual}% too far from target {test_speed}%"
        print(f"Fan {fan_id} verified at {actual}%")

    release_fan_control(gpu_id, args.display, verbose=args.verbose)
    print("Test passed!")


def main(argv=None):
    args = parse_args(argv)

    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    buses = gpu_buses(verbose=args.verbose)

    with managed_xserver(args.display, kill=args.kill, verbose=args.verbose):
        # Discover fan-to-GPU mapping
        gpu_to_fans = discover_fan_to_gpu_map(args.display, verbose=args.verbose)
        print(f"Discovered GPU-to-fan mapping: {gpu_to_fans}")

        # Get speed ranges for all fans
        all_fan_ids = [fid for fids in gpu_to_fans.values() for fid in fids]
        fan_speed_ranges = get_fan_speed_ranges(all_fan_ids, args.display, verbose=args.verbose)

        if args.test:
            test_mode(args, buses, gpu_to_fans, fan_speed_ranges)
        else:
            manage_fans(args, buses, gpu_to_fans, fan_speed_ranges)
```

- [ ] **Step 8: Verify the refactor builds and installs**

```bash
pip install -e ".[dev]"
coolgpus --help
```

Expected: help text prints with the new `--interval` argument visible.

- [ ] **Step 9: Delete old files**

Remove `setup.py` and the old `coolgpus` script file now that the package structure is in place.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml src/
git rm setup.py coolgpus
git commit -m "refactor: convert to package structure with proper module separation

Split monolithic script into cli, core, nvidia, and xserver modules.
Add pyproject.toml, --interval flag, signal handling, and fan-to-GPU discovery."
```

---

## Chunk 2: Unit Tests for Core Logic

### Task 2: Test the pure fan curve logic in core.py

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_core.py`

These tests cover the pure math functions — no mocking needed.

- [ ] **Step 1: Create `tests/__init__.py`**

Empty file.

- [ ] **Step 2: Write tests for `clamp()`**

```python
from coolgpus.core import clamp


def test_clamp_within_range():
    assert clamp(50, 0, 100) == 50


def test_clamp_below_min():
    assert clamp(-5, 0, 100) == 0


def test_clamp_above_max():
    assert clamp(150, 0, 100) == 100


def test_clamp_at_boundaries():
    assert clamp(0, 0, 100) == 0
    assert clamp(100, 0, 100) == 100
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_core.py -v
```

Expected: all pass.

- [ ] **Step 4: Write tests for `determine_segment()`**

```python
from coolgpus.core import determine_segment


def test_determine_segment_below_min():
    result = determine_segment(10, [55, 80], [30, 99])
    assert result == (55, 80, 30, 99)


def test_determine_segment_in_range():
    result = determine_segment(65, [55, 80], [30, 99])
    assert result == (55, 80, 30, 99)


def test_determine_segment_above_max():
    result = determine_segment(90, [55, 80], [30, 99])
    assert result == (55, 80, 30, 99)


def test_determine_segment_multi_segment():
    temps = [20, 55, 80]
    speeds = [5, 30, 99]
    assert determine_segment(10, temps, speeds) == (20, 55, 5, 30)
    assert determine_segment(40, temps, speeds) == (20, 55, 5, 30)
    assert determine_segment(60, temps, speeds) == (55, 80, 30, 99)
    assert determine_segment(90, temps, speeds) == (55, 80, 30, 99)
```

- [ ] **Step 5: Write tests for `min_speed()` and `target_speed()`**

```python
from coolgpus.core import min_speed, target_speed


def test_min_speed_at_low_temp():
    assert min_speed(50, [55, 80], [30, 99]) == 30


def test_min_speed_at_high_temp():
    assert min_speed(85, [55, 80], [30, 99]) == 99


def test_min_speed_midpoint():
    # At 67.5C (midpoint of 55-80), speed should be midpoint of 30-99 = 64
    result = min_speed(67.5, [55, 80], [30, 99])
    assert result == 64


def test_target_speed_ramps_up():
    # Current speed 0, temp 70 -> should increase
    s, lo, hi = target_speed(0, 70, [55, 80], [30, 99], 2)
    assert s == lo  # speed should be at least the minimum


def test_target_speed_hysteresis():
    # Current speed 80, temp drops slightly -> should stay at 80 due to hysteresis
    s, lo, hi = target_speed(80, 68, [55, 80], [30, 99], 2)
    assert lo <= s <= hi
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/test_core.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/
git commit -m "test: add unit tests for core fan curve logic"
```

---

## Chunk 3: Systemd Service File and README Update

### Task 3: Ship a systemd service file

**Files:**
- Create: `coolgpus.service`

- [ ] **Step 1: Create `coolgpus.service`**

```ini
[Unit]
Description=Headless GPU Fan Control
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
ExecStart=/usr/local/bin/coolgpus --kill
Restart=on-failure
RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Commit**

```bash
git add coolgpus.service
git commit -m "feat: add systemd service file"
```

### Task 4: Update README

**Files:**
- Modify: `readme.md`

- [ ] **Step 1: Update readme.md**

Key changes:
- Remove abandonware notice (first 3 lines)
- Update the nvidia-smi example output to a more modern driver version
- Add note about multi-fan GPU support
- Add note about `--interval` flag
- Update systemd section to reference the shipped service file
- Update credit section to mention this fork
- Add note about Ubuntu 24.04 compatibility
- Add driver compatibility warnings (560.35.03, 565, 570 known issues)

- [ ] **Step 2: Commit**

```bash
git add readme.md
git commit -m "docs: update README for 2026, remove abandonware notice"
```

---

## Chunk 4: Driver Compatibility Notes

### Task 5: Add driver version detection and warnings

**Files:**
- Modify: `src/coolgpus/nvidia.py`
- Modify: `src/coolgpus/cli.py`

Research has identified these driver versions with known fan control issues:
- Driver 560.35.03: XNVCtrl fan control broken (BadValue errors)
- Driver 565.x: Fans may not appear in nvidia-settings
- Driver 570.x (open kernel): Fans stuck at 30%

- [ ] **Step 1: Add `driver_version()` to `nvidia.py`**

```python
def driver_version(verbose=False):
    """Get the NVIDIA driver version string."""
    output = log_output(
        ["nvidia-smi", "--format=csv,noheader", "--query-gpu=driver_version"],
        verbose=verbose,
    )
    return output.splitlines()[0].strip()
```

- [ ] **Step 2: Add driver check to `cli.py` `main()`**

After getting the driver version, print a warning if it matches a known problematic version. Don't block execution — just warn.

```python
KNOWN_PROBLEMATIC_DRIVERS = {
    "560.35.03": "XNVCtrl fan control may not work (BadValue errors)",
    "565": "Fans may not appear in nvidia-settings",
    "570": "Fans may get stuck at 30% with open kernel module",
}

def check_driver(verbose=False):
    version = driver_version(verbose=verbose)
    print(f"NVIDIA driver version: {version}")
    for prefix, warning in KNOWN_PROBLEMATIC_DRIVERS.items():
        if version.startswith(prefix):
            print(f"WARNING: Driver {version} has known issues: {warning}")
            print("See https://github.com/tazitoo/coolgpus_2026 for details.")
    return version
```

- [ ] **Step 3: Call `check_driver()` at the start of `main()`**

- [ ] **Step 4: Commit**

```bash
git add src/coolgpus/nvidia.py src/coolgpus/cli.py
git commit -m "feat: add driver version detection with known-issue warnings"
```
