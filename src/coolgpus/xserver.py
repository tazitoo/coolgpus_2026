import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from subprocess import DEVNULL, Popen

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
    # Use the path that was used to invoke this script, since shutil.which
    # won't find it under sudo (root has a different PATH)
    coolgpus_path = shutil.which("coolgpus") or str(Path(sys.argv[0]).resolve())
    if not Path(coolgpus_path).is_file():
        print(f"WARNING: could not find coolgpus executable, skipping service install.")
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


def start_xserver(display, verbose=False):
    """Start an X server on the given display with no access control."""
    # Clear XAUTHORITY so nvidia-settings doesn't try to authenticate
    os.environ.pop("XAUTHORITY", None)
    xorgargs = ["Xorg", display, "-once", "-ac"]
    print("Starting xserver: " + " ".join(xorgargs))
    p = Popen(xorgargs)
    time.sleep(2)  # give X server time to initialize
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


def is_alive(process):
    """Check if an X server process is still running."""
    return process is not None and process.poll() is None


@contextmanager
def managed_xserver(display, kill=False, verbose=False):
    """Context manager that starts an X server and cleans up on exit.

    Yields a health-check function that the main loop can call to detect
    and recover from Xorg crashes.
    """
    kill_xservers(kill=kill, verbose=verbose)
    xserver_process = start_xserver(display, verbose=verbose)

    def check_and_restart():
        """Returns True if Xorg is healthy. Restarts it if dead."""
        nonlocal xserver_process
        if is_alive(xserver_process):
            return True
        print("WARNING: Xorg server died. Restarting...")
        xserver_process = start_xserver(display, verbose=verbose)
        time.sleep(2)  # give it a moment to initialize
        if is_alive(xserver_process):
            print("Xorg server restarted successfully.")
            return True
        else:
            print("ERROR: Xorg server failed to restart.")
            return False

    try:
        yield check_and_restart
    finally:
        if is_alive(xserver_process):
            print(f"Terminating xserver for display {display}.")
            xserver_process.terminate()
