import argparse
import signal
import sys
import time

from coolgpus.core import clamp, target_speed
from coolgpus.nvidia import (
    discover_fan_to_gpu_map,
    driver_version,
    fetch_current_fan_speed,
    get_fan_speed_ranges,
    gpu_buses,
    release_fan_control,
    set_fan_speed,
    temperature,
)
from coolgpus.xserver import managed_xserver

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
    parser.add_argument("--display", type=str, default=":0", help="X server display to use (default: :0)")
    parser.add_argument("--test", action="store_true", help="Run hardware test mode")
    args = parser.parse_args(argv)

    assert len(args.temp) == len(args.speed), "temp and speed should have the same length"
    assert len(args.temp) >= 2, "Please use at least two points for temp"
    assert len(args.speed) >= 2, "Please use at least two points for speed"

    return args


def manage_fans(args, buses, gpu_to_fans, fan_speed_ranges):
    """Main fan control loop. Adjusts fan speeds based on GPU temperatures."""
    speeds = {gpu_id: 0 for gpu_id in gpu_to_fans}

    try:
        while True:
            start_time = time.time()

            for gpu_id, fan_ids in gpu_to_fans.items():
                bus = buses[gpu_id]
                temp = temperature(bus, verbose=args.verbose)

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

    gpu_count = len(buses)
    assert len(gpu_to_fans) == gpu_count, f"GPU count mismatch: {len(gpu_to_fans)} vs {gpu_count}"

    for fan_id, (lo, hi) in fan_speed_ranges.items():
        assert 0 <= lo <= 45, f"Fan {fan_id} min speed {lo} outside expected range 0-45"
        assert 75 <= hi <= 110, f"Fan {fan_id} max speed {hi} outside expected range 75-110"

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

    for fan_id in fan_ids:
        actual = fetch_current_fan_speed(fan_id, args.display, verbose=args.verbose)
        assert actual > test_speed - 10, f"Fan {fan_id} speed {actual}% too far from target {test_speed}%"
        print(f"Fan {fan_id} verified at {actual}%")

    release_fan_control(gpu_id, args.display, verbose=args.verbose)
    print("Test passed!")


def main(argv=None):
    args = parse_args(argv)

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    buses = gpu_buses(verbose=args.verbose)
    check_driver(verbose=args.verbose)

    with managed_xserver(args.display, kill=args.kill, verbose=args.verbose):
        gpu_to_fans = discover_fan_to_gpu_map(args.display, verbose=args.verbose)
        print(f"Discovered GPU-to-fan mapping: {gpu_to_fans}")

        all_fan_ids = [fid for fids in gpu_to_fans.values() for fid in fids]
        fan_speed_ranges = get_fan_speed_ranges(all_fan_ids, args.display, verbose=args.verbose)

        if args.test:
            test_mode(args, buses, gpu_to_fans, fan_speed_ranges)
        else:
            manage_fans(args, buses, gpu_to_fans, fan_speed_ranges)
