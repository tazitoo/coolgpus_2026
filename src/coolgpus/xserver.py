import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from subprocess import DEVNULL, Popen
from tempfile import mkdtemp

from coolgpus.nvidia import log_output

SERVICE_TEMPLATE = """\
[Unit]
Description=Headless GPU Fan Control
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
Environment="PATH={bin_dir}:/usr/local/bin:/usr/bin:/bin"
ExecStart={coolgpus_path} --kill
Restart=on-failure
RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
"""

# Per-GPU xorg.conf template — one screen, one device, bound to a specific BusID
XORG_CONF_TEMPLATE = """\
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"     0    0
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "VideoCard0"
    Monitor        "Monitor0"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    Option         "Coolbits" "4"
    SubSection "Display"
        Depth       24
    EndSubSection
EndSection

Section "ServerFlags"
    Option         "AllowEmptyInput" "on"
EndSection

Section "Device"
    Identifier  "VideoCard0"
    Driver      "nvidia"
    Option      "Coolbits" "4"
    BusID       "PCI:{bus}"
EndSection

Section "Monitor"
    Identifier      "Monitor0"
    VendorName      "Dummy Display"
    ModelName       "640x480"
EndSection
"""


def decimalize(bus):
    """Convert a PCI bus ID like '00000000:81:00.0' to Xorg format like '129:0:0'."""
    # Drop the domain prefix (first 9 chars: '00000000:'), then convert hex parts
    return ":".join(str(int(p, 16)) for p in re.split("[:.]", bus[9:]))


def configure_xorg(verbose=False):
    """Update /etc/X11/xorg.conf with cool-bits enabled for fan control,
    and install the systemd service.

    Requires root. Only needs to be run once (or after driver updates).
    """
    log_output(
        [
            "nvidia-xconfig",
            "--enable-all-gpus",
            "--cool-bits=4",
            "--allow-empty-initial-configuration",
        ],
        verbose=verbose,
    )
    print("Updated /etc/X11/xorg.conf with cool-bits=4 for all GPUs.")

    install_service(verbose=verbose)


def install_service(verbose=False):
    """Generate and install the systemd service file using the current coolgpus path."""
    coolgpus_path = shutil.which("coolgpus") or str(Path(sys.argv[0]).resolve())
    if not Path(coolgpus_path).is_file():
        print("WARNING: could not find coolgpus executable, skipping service install.")
        return

    bin_dir = str(Path(coolgpus_path).parent)
    service_content = SERVICE_TEMPLATE.format(
        bin_dir=bin_dir,
        coolgpus_path=coolgpus_path,
    )

    service_path = Path("/etc/systemd/system/coolgpus.service")
    service_path.write_text(service_content)
    print(f"Installed systemd service to {service_path}")
    print(f"  Using executable: {coolgpus_path}")

    log_output(["systemctl", "daemon-reload"], verbose=verbose)
    log_output(["systemctl", "enable", "coolgpus"], verbose=verbose)
    print("Service enabled. Start with: sudo systemctl start coolgpus")


def _gen_gpu_config(bus):
    """Generate a per-GPU xorg.conf in a temp directory, returns config path."""
    tempdir = mkdtemp(prefix="coolgpus-" + bus.replace(":", "-"))
    conf = os.path.join(tempdir, "xorg.conf")
    with open(conf, "w") as f:
        f.write(XORG_CONF_TEMPLATE.format(bus=decimalize(bus)))
    return conf


def _clean_stale_lock(display, verbose=False):
    """Remove stale X lock files left by a previous crash."""
    display_num = display.lstrip(":")
    lock_file = Path(f"/tmp/.X{display_num}-lock")
    socket_file = Path(f"/tmp/.X11-unix/X{display_num}")
    for f in (lock_file, socket_file):
        if f.exists():
            if verbose:
                print(f"Removing stale lock file: {f}")
            f.unlink()


def start_xserver(display, bus, verbose=False):
    """Start an X server for a single GPU on the given display."""
    os.environ.pop("XAUTHORITY", None)
    _clean_stale_lock(display, verbose=verbose)
    conf = _gen_gpu_config(bus)
    xorgargs = ["Xorg", display, "-ac", "-noreset", "-config", conf]
    print(f"Starting xserver for GPU {bus}: " + " ".join(xorgargs))
    p = Popen(xorgargs, stdout=DEVNULL, stderr=DEVNULL)
    time.sleep(2)
    if verbose:
        print(f"Started xserver on {display} for {bus}")
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


def is_alive(process):
    """Check if an X server process is still running."""
    return process is not None and process.poll() is None


@contextmanager
def managed_xservers(buses, base_display=8, kill=False, verbose=False):
    """Context manager that starts a single shared X server for all GPUs.

    Uses the system /etc/X11/xorg.conf (configured by --setup with cool-bits
    for all GPUs) so that one Xorg instance can control fans on every GPU.

    Yields:
        display: display string (e.g. ':8')
        check_fn: health-check function that restarts dead Xorg process
    """
    kill_xservers(kill=kill, verbose=verbose)

    display = f":{base_display}"
    _clean_stale_lock(display, verbose=verbose)

    xorgargs = ["Xorg", display, "-ac", "-noreset"]
    print(f"Starting shared xserver: " + " ".join(xorgargs))
    process = Popen(xorgargs, stdout=DEVNULL, stderr=DEVNULL)
    time.sleep(2)

    def check_and_restart():
        nonlocal process
        if is_alive(process):
            return True
        print(f"WARNING: Xorg ({display}) died. Restarting...")
        _clean_stale_lock(display, verbose=verbose)
        process = Popen(xorgargs, stdout=DEVNULL, stderr=DEVNULL)
        time.sleep(2)
        if is_alive(process):
            print(f"Xorg restarted successfully.")
            return True
        print(f"ERROR: Xorg failed to restart.")
        return False

    try:
        yield display, check_and_restart
    finally:
        if is_alive(process):
            print(f"Terminating xserver ({display}).")
            process.terminate()
