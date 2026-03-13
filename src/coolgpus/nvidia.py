import re
import time
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
    """Get all fan indices visible on this display.

    Returns all fan indices. Fan indices are global, but only the fans
    belonging to this display's GPU will respond to speed commands —
    the rest are harmlessly ignored.
    """
    output = log_output(
        ["nvidia-settings", "-q", "fans", "-c", display],
        verbose=verbose,
    )
    return [int(m) for m in re.findall(r"\[fan:(\d+)\]", output)]


def build_gpu_bus_map(display, buses, verbose=False):
    """Map nvidia-smi GPU indices to nvidia-settings gpu:N indices via PCIBus.

    Returns dict of smi_idx -> ns_idx.
    """
    bus_to_smi = {}
    for smi_idx, bus in enumerate(buses):
        bus_num = int(bus[9:].split(":")[0], 16)
        bus_to_smi[bus_num] = smi_idx

    smi_to_ns = {}
    for ns_idx in range(len(buses)):
        pci_bus = int(log_output(
            ["nvidia-settings", "-q", f"[gpu:{ns_idx}]/PCIBus", "-c", display, "--terse"],
            verbose=verbose,
            stderr=DEVNULL,
        ))
        if pci_bus in bus_to_smi:
            smi_to_ns[bus_to_smi[pci_bus]] = ns_idx
            if verbose:
                print(f"nvidia-smi GPU {bus_to_smi[pci_bus]} -> nvidia-settings gpu:{ns_idx} (PCIBus {pci_bus})")

    return smi_to_ns


def probe_fan_mapping(display, fan_indices, num_gpus, verbose=False):
    """Probe which fans each nvidia-settings gpu:N controls.

    Enables one GPU's fan control at a time, sets all fans to max,
    and checks which fans respond with increased RPM.

    Returns dict of ns_gpu_idx -> list of fan indices.
    """
    PROBE_SPEED = 99
    SETTLE_TIME = 5
    RPM_INCREASE_THRESHOLD = 500

    # Zero all fans first to get a clean baseline (fans may be spinning
    # from a previous session or manual testing)
    for g in range(num_gpus):
        log_output(
            ["nvidia-settings", "-a", f"[gpu:{g}]/GPUFanControlState=1", "-c", display],
            verbose=verbose,
        )
    for fan_id in fan_indices:
        log_output(
            ["nvidia-settings", "-a", f"[fan:{fan_id}]/GPUTargetFanSpeed=0", "-c", display],
            verbose=verbose,
        )
    time.sleep(5)

    # Disable all fan control, get baseline RPMs
    for g in range(num_gpus):
        log_output(
            ["nvidia-settings", "-a", f"[gpu:{g}]/GPUFanControlState=0", "-c", display],
            verbose=verbose,
        )
    time.sleep(3)

    baseline = {}
    for fan_id in fan_indices:
        baseline[fan_id] = int(log_output(
            ["nvidia-settings", "-q", f"[fan:{fan_id}]/GPUCurrentFanSpeedRPM", "-c", display, "--terse"],
            verbose=verbose,
            stderr=DEVNULL,
        ))
    print(f"Fan probe baseline RPMs: {baseline}")

    mapping = {}
    for gpu_idx in range(num_gpus):
        log_output(
            ["nvidia-settings", "-a", f"[gpu:{gpu_idx}]/GPUFanControlState=1", "-c", display],
            verbose=verbose,
        )

        for fan_id in fan_indices:
            log_output(
                ["nvidia-settings", "-a", f"[fan:{fan_id}]/GPUTargetFanSpeed={PROBE_SPEED}", "-c", display],
                verbose=verbose,
            )

        time.sleep(SETTLE_TIME)

        responding = []
        for fan_id in fan_indices:
            rpm = int(log_output(
                ["nvidia-settings", "-q", f"[fan:{fan_id}]/GPUCurrentFanSpeedRPM", "-c", display, "--terse"],
                verbose=verbose,
                stderr=DEVNULL,
            ))
            if rpm - baseline.get(fan_id, 0) > RPM_INCREASE_THRESHOLD:
                responding.append(fan_id)
                if verbose:
                    print(f"  fan:{fan_id}: {baseline[fan_id]} -> {rpm} RPM -> gpu:{gpu_idx}")

        mapping[gpu_idx] = responding
        print(f"nvidia-settings gpu:{gpu_idx}: fans {responding}")

        log_output(
            ["nvidia-settings", "-a", f"[gpu:{gpu_idx}]/GPUFanControlState=0", "-c", display],
            verbose=verbose,
        )
        time.sleep(2)

    return mapping


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


def set_fan_speed(fan_ids, target, display, gpu_ns_idx=0, verbose=False):
    """Enable manual fan control and set fans to target speed."""
    log_output(
        ["nvidia-settings", "-a", f"[gpu:{gpu_ns_idx}]/GPUFanControlState=1", "-c", display],
        verbose=verbose,
    )
    for fan_id in fan_ids:
        log_output(
            ["nvidia-settings", "-a", f"[fan:{fan_id}]/GPUTargetFanSpeed={int(target)}", "-c", display],
            verbose=verbose,
        )


def release_fan_control(display, gpu_ns_idx=0, verbose=False):
    """Return fan control to the driver."""
    log_output(
        ["nvidia-settings", "-a", f"[gpu:{gpu_ns_idx}]/GPUFanControlState=0", "-c", display],
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
