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
