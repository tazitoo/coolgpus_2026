import re
from subprocess import TimeoutExpired, Popen, PIPE, STDOUT, DEVNULL


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
    """Get the hottest temperature on the GPU (max of die and VRAM).

    Falls back to die-only if VRAM temp is not available (reported as
    'N/A' or '[Not Supported]' on some cards).
    """
    output = log_output(
        ["nvidia-smi", "--format=csv,noheader,nounits",
         "--query-gpu=temperature.gpu,temperature.memory", "-i", bus],
        verbose=verbose,
    )
    parts = [p.strip() for p in output.splitlines()[0].split(",")]
    die_temp = int(parts[0])
    try:
        mem_temp = int(parts[1])
    except (ValueError, IndexError):
        mem_temp = 0
    temp = max(die_temp, mem_temp)
    if verbose and mem_temp > 0:
        print(f"  GPU die: {die_temp}C, VRAM: {mem_temp}C -> using {temp}C")
    return temp


def discover_fans(display, verbose=False):
    """Query nvidia-settings on a per-GPU display to find its fans.

    Each Xorg server is bound to a single GPU, so all fans belong to gpu:0.
    Returns list of fan indices (e.g. [0, 1]).
    """
    output = log_output(
        ["nvidia-settings", "-q", "fans", "-c", display],
        verbose=verbose,
    )
    return [int(m) for m in re.findall(r"\[fan:(\d+)\]", output)]


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


def set_fan_speed(fan_ids, target, display, verbose=False):
    """Enable manual fan control and set all fans to target speed.

    Each display is bound to a single GPU, so we always use [gpu:0].
    """
    log_output(
        ["nvidia-settings", "-a", "[gpu:0]/GPUFanControlState=1", "-c", display],
        verbose=verbose,
    )
    for fan_id in fan_ids:
        log_output(
            ["nvidia-settings", "-a", f"[fan:{fan_id}]/GPUTargetFanSpeed={int(target)}", "-c", display],
            verbose=verbose,
        )


def release_fan_control(display, verbose=False):
    """Return fan control to the driver. Each display has one GPU (gpu:0)."""
    log_output(
        ["nvidia-settings", "-a", "[gpu:0]/GPUFanControlState=0", "-c", display],
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


def get_power_limits(bus, verbose=False):
    """Get default and current power limits for a GPU in watts.

    Returns (default_limit, current_limit).
    """
    default = log_output(
        ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=power.default_limit", "-i", bus],
        verbose=verbose,
    ).strip()
    current = log_output(
        ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=power.limit", "-i", bus],
        verbose=verbose,
    ).strip()
    return float(default), float(current)


def set_power_limit(bus, watts, verbose=False):
    """Set GPU power limit in watts via nvidia-smi."""
    log_output(
        ["nvidia-smi", "-i", bus, "-pl", str(int(watts))],
        verbose=verbose,
    )


def driver_version(verbose=False):
    """Get the NVIDIA driver version string."""
    output = log_output(
        ["nvidia-smi", "--format=csv,noheader", "--query-gpu=driver_version"],
        verbose=verbose,
    )
    return output.splitlines()[0].strip()
