from contextlib import contextmanager
import pynvml

_NVML_TEMPERATURE_GPU = 0
_NVML_TEMPERATURE_MEMORY = 1  # supported on some cards/drivers; falls back gracefully


@contextmanager
def nvml_context():
    """Initialize NVML on enter, shut down on exit. Wraps the main loop."""
    pynvml.nvmlInit()
    try:
        yield
    finally:
        pynvml.nvmlShutdown()


def gpu_count(verbose=False):
    """Return the number of NVIDIA GPUs visible to NVML."""
    count = pynvml.nvmlDeviceGetCount()
    if verbose:
        print(f"Found {count} GPU(s)")
    return count


def temperature(gpu_id, verbose=False):
    """Return the hottest temperature on a GPU (max of die and VRAM).

    Falls back to die-only if the VRAM sensor is not available on this card.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    die_temp = pynvml.nvmlDeviceGetTemperature(handle, _NVML_TEMPERATURE_GPU)
    try:
        mem_temp = pynvml.nvmlDeviceGetTemperature(handle, _NVML_TEMPERATURE_MEMORY)
    except pynvml.NVMLError:
        mem_temp = 0
    temp = max(die_temp, mem_temp)
    if verbose and mem_temp > 0:
        print(f"  GPU {gpu_id} die: {die_temp}C, VRAM: {mem_temp}C -> using {temp}C")
    return temp


def discover_fan_to_gpu_map(verbose=False):
    """Return a dict mapping gpu_id (int) -> list of fan indices for that GPU."""
    gpu_to_fans = {}
    count = pynvml.nvmlDeviceGetCount()
    for gpu_id in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        num_fans = pynvml.nvmlDeviceGetNumFans(handle)
        gpu_to_fans[gpu_id] = list(range(num_fans))
        if verbose:
            print(f"  GPU {gpu_id}: {num_fans} fan(s)")
    return gpu_to_fans


def get_fan_speed_ranges(gpu_to_fans, verbose=False):
    """Return valid speed range per GPU as {gpu_id: (min%, max%)}.

    Tries nvmlDeviceGetMinMaxFanSpeed (driver >= 520); falls back to (0, 100).
    """
    ranges = {}
    for gpu_id in gpu_to_fans:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        try:
            lo, hi = pynvml.nvmlDeviceGetMinMaxFanSpeed(handle)
        except pynvml.NVMLError:
            lo, hi = 0, 100
            if verbose:
                print(f"  GPU {gpu_id}: min/max fan speed not available, using 0-100%")
        ranges[gpu_id] = (lo, hi)
        if verbose:
            print(f"  GPU {gpu_id} fan range: {lo}-{hi}%")
    return ranges


def set_fan_speed(gpu_id, fan_ids, target, verbose=False):
    """Set all fans on a GPU to target speed (percent)."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    for fan_id in fan_ids:
        pynvml.nvmlDeviceSetFanSpeed_v2(handle, fan_id, int(target))
    if verbose:
        print(f"  GPU {gpu_id} fans {fan_ids} -> {int(target)}%")


def release_fan_control(gpu_id, verbose=False):
    """Return fan control to the driver for a GPU."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    num_fans = pynvml.nvmlDeviceGetNumFans(handle)
    for fan_id in range(num_fans):
        pynvml.nvmlDeviceSetDefaultFanSpeed_v2(handle, fan_id)
    if verbose:
        print(f"  Released fan control for GPU {gpu_id}")


def fetch_current_fan_speed(gpu_id, fan_id, verbose=False):
    """Return the current fan speed percentage for a specific fan."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    speed = pynvml.nvmlDeviceGetFanSpeed_v2(handle, fan_id)
    if verbose:
        print(f"  GPU {gpu_id} fan {fan_id}: {speed}%")
    return speed


def get_power_limits(gpu_id, verbose=False):
    """Return (default_watts, current_watts) power limits for a GPU.

    NVML reports milliwatts; this function converts to watts.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    default_w = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
    current_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    return default_w, current_w


def set_power_limit(gpu_id, watts, verbose=False):
    """Set GPU power limit. Converts watts to milliwatts for NVML."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(watts * 1000))
    if verbose:
        print(f"  GPU {gpu_id} power limit -> {watts}W")


def driver_version(verbose=False):
    """Return the NVIDIA driver version string."""
    version = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(version, bytes):
        version = version.decode()
    return version
