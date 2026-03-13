import argparse
import signal
import sys
import time

from coolgpus.core import clamp, target_speed
from coolgpus.nvidia import (
    discover_fans,
    driver_version,
    fetch_current_fan_speed,
    get_fan_speed_ranges,
    get_power_limits,
    gpu_buses,
    release_fan_control,
    set_fan_speed,
    set_power_limit,
    temperature,
)
from coolgpus.xserver import configure_xorg, managed_xservers

KNOWN_PROBLEMATIC_DRIVERS = {
    "560.35.03": "XNVCtrl fan control may not work (BadValue errors)",
    "565": "Fans may not appear in nvidia-settings",
    "570": "Fans may get stuck at 30% with open kernel module",
}


def check_driver(verbose=False):
    """Check driver version and warn about known issues."""
    version = driver_version(verbose=verbose)
    print(f"NVIDIA driver version: {version}")
    for prefix, warning in KNOWN_PROBLEMATIC_DRIVERS.items():
        if version.startswith(prefix):
            print(f"WARNING: Driver {version} has known issues: {warning}")
            print("See https://github.com/tazitoo/coolgpus_2026 for details.")
    return version


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="""\
GPU fan control for headless Linux.

By default, uses a clamped linear fan curve from 30% below 55C to 99% above 80C,
with a small hysteresis gap to reduce fan speed oscillation."""
    )
    parser.add_argument(
        "--temp", nargs="+", default=[40, 80], type=float,
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
    parser.add_argument("--kill", action=argparse.BooleanOptionalAction, default=True, help="Kill existing Xorg sessions before starting")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print extra debugging information")
    parser.add_argument("--base-display", type=int, default=8, help="Starting X display number (default: 8, uses 8,9,... for each GPU)")
    parser.add_argument("--max-temp", type=float, default=75,
        help="Temperature ceiling in celsius. If exceeded, power limit is reduced (default: 75)")
    parser.add_argument("--power-step", type=float, default=25,
        help="Watts to reduce power limit per step when over max-temp (default: 25)")
    parser.add_argument("--setup", action="store_true", help="Write /etc/X11/xorg.conf with cool-bits enabled (requires root, run once)")
    parser.add_argument("--test", action="store_true", help="Run hardware test mode")
    args = parser.parse_args(argv)

    assert len(args.temp) == len(args.speed), "temp and speed should have the same length"
    assert len(args.temp) >= 2, "Please use at least two points for temp"
    assert len(args.speed) >= 2, "Please use at least two points for speed"

    return args


def manage_fans(args, buses, gpu_fans, gpu_displays, xorg_check):
    """Main fan control loop. Adjusts fan speeds based on GPU temperatures.

    gpu_fans: dict mapping gpu_index -> list of fan_indices
    gpu_displays: dict mapping gpu_index -> display string
    """
    speeds = {gpu_idx: 0 for gpu_idx in gpu_fans}

    fan_speed_ranges = {}
    for gpu_idx, fan_ids in gpu_fans.items():
        display = gpu_displays[gpu_idx]
        fan_speed_ranges[gpu_idx] = get_fan_speed_ranges(fan_ids, display, verbose=args.verbose)

    power_state = {}
    for gpu_idx in gpu_fans:
        bus = buses[gpu_idx]
        default_pl, current_pl = get_power_limits(bus, verbose=args.verbose)
        power_state[gpu_idx] = {"default": default_pl, "current": current_pl, "floor": default_pl * 0.33}
        print(f"GPU {gpu_idx} power limit: {current_pl}W (default: {default_pl}W, floor: {default_pl * 0.33:.0f}W)")

    try:
        while True:
            start_time = time.time()

            if not xorg_check():
                print("ERROR: Cannot recover Xorg. Exiting.")
                break

            for gpu_idx, fan_ids in gpu_fans.items():
                bus = buses[gpu_idx]
                display = gpu_displays[gpu_idx]
                temp = temperature(bus, verbose=args.verbose)

                ranges = fan_speed_ranges[gpu_idx]
                fan_min, fan_max = ranges[fan_ids[0]]

                s, lo, hi = target_speed(speeds[gpu_idx], temp, args.temp, args.speed, args.hyst)
                s = clamp(s, fan_min, fan_max)
                lo = clamp(lo, fan_min, fan_max)
                hi = clamp(hi, fan_min, fan_max)

                if s != speeds[gpu_idx]:
                    print(f"GPU {gpu_idx}, {temp}C -> [{lo}%-{hi}%]. Setting speed to {s}%")
                    set_fan_speed(fan_ids, s, display, verbose=args.verbose)
                elif args.verbose:
                    print(f"GPU {gpu_idx}, {temp}C -> [{lo}%-{hi}%]. Leaving speed at {s}%")

                speeds[gpu_idx] = s

                ps = power_state[gpu_idx]
                if temp > args.max_temp:
                    new_pl = ps["current"] - args.power_step
                    if new_pl >= ps["floor"]:
                        print(f"WARNING: GPU {gpu_idx} at {temp}C (>{args.max_temp}C). "
                              f"Reducing power limit: {ps['current']:.0f}W -> {new_pl:.0f}W")
                        set_power_limit(bus, new_pl, verbose=args.verbose)
                        ps["current"] = new_pl
                    else:
                        print(f"WARNING: GPU {gpu_idx} at {temp}C. Power already at floor "
                              f"({ps['current']:.0f}W). Cannot reduce further.")
                elif temp < args.max_temp - args.hyst and ps["current"] < ps["default"]:
                    new_pl = min(ps["current"] + args.power_step, ps["default"])
                    print(f"GPU {gpu_idx} cooled to {temp}C. "
                          f"Restoring power limit: {ps['current']:.0f}W -> {new_pl:.0f}W")
                    set_power_limit(bus, new_pl, verbose=args.verbose)
                    ps["current"] = new_pl

            elapsed = time.time() - start_time
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
    finally:
        for gpu_idx in gpu_fans:
            try:
                bus = buses[gpu_idx]
                display = gpu_displays[gpu_idx]
                ps = power_state[gpu_idx]
                if ps["current"] < ps["default"]:
                    print(f"Restoring GPU {gpu_idx} power limit to {ps['default']:.0f}W")
                    set_power_limit(bus, ps["default"], verbose=args.verbose)
                release_fan_control(display, verbose=args.verbose)
                print(f"Released fan speed control for GPU {gpu_idx}")
            except Exception as e:
                print(f"Cleanup error for GPU {gpu_idx} (Xorg may be gone): {e}")


def test_mode(args, buses, gpu_fans, gpu_displays):
    """Run hardware validation tests."""
    import random

    gpu_count = len(buses)
    assert len(gpu_fans) == gpu_count, f"GPU count mismatch: {len(gpu_fans)} vs {gpu_count}"

    gpu_idx = random.choice(list(gpu_fans.keys()))
    bus = buses[gpu_idx]
    display = gpu_displays[gpu_idx]
    fan_ids = gpu_fans[gpu_idx]

    fan_speed_ranges = get_fan_speed_ranges(fan_ids, display, verbose=args.verbose)
    fan_min, fan_max = fan_speed_ranges[fan_ids[0]]

    temp = temperature(bus, verbose=args.verbose)
    assert 0 <= temp <= 100, f"GPU {gpu_idx} temperature {temp}C outside expected range"

    test_speed = clamp(90, fan_min, fan_max)
    print(f"Setting GPU {gpu_idx} fans to {test_speed}%...")
    set_fan_speed(fan_ids, test_speed, display, verbose=args.verbose)
    time.sleep(20)

    for fan_id in fan_ids:
        actual = fetch_current_fan_speed(fan_id, display, verbose=args.verbose)
        assert actual > test_speed - 10, f"Fan {fan_id} speed {actual}% too far from target {test_speed}%"
        print(f"Fan {fan_id} verified at {actual}%")

    release_fan_control(display, verbose=args.verbose)
    print("Test passed!")


def main(argv=None):
    # Ensure print output appears immediately in systemd journal
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    args = parse_args(argv)

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    buses = gpu_buses(verbose=args.verbose)
    check_driver(verbose=args.verbose)

    if args.setup:
        configure_xorg(verbose=args.verbose)
        return

    with managed_xservers(buses, base_display=args.base_display,
                          kill=args.kill, verbose=args.verbose) as (gpu_displays, xorg_check):
        # Discover fans on each per-GPU Xorg server
        gpu_fans = {}
        for gpu_idx, display in gpu_displays.items():
            for attempt in range(5):
                fans = discover_fans(display, verbose=args.verbose)
                if fans:
                    gpu_fans[gpu_idx] = fans
                    break
                print(f"Fan discovery for GPU {gpu_idx} ({display}) attempt {attempt + 1}/5, retrying in 3s...")
                time.sleep(3)
            if gpu_idx not in gpu_fans:
                raise RuntimeError(f"Could not discover fans for GPU {gpu_idx} on {display}")

        for gpu_idx, fan_ids in gpu_fans.items():
            print(f"GPU {gpu_idx} ({buses[gpu_idx]}) on {gpu_displays[gpu_idx]}: fans {fan_ids}")

        if args.test:
            test_mode(args, buses, gpu_fans, gpu_displays)
        else:
            manage_fans(args, buses, gpu_fans, gpu_displays, xorg_check)
