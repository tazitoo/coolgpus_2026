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
    output = log_output(
        ["nvidia-settings", "-q", "fans", "-c", display],
        verbose=verbose,
    )
    fan_indices = [int(m) for m in re.findall(r"\[fan:(\d+)\]", output)]

    gpu_to_fans = {}
    for fan_id in fan_indices:
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


def driver_version(verbose=False):
    """Get the NVIDIA driver version string."""
    output = log_output(
        ["nvidia-smi", "--format=csv,noheader", "--query-gpu=driver_version"],
        verbose=verbose,
    )
    return output.splitlines()[0].strip()
