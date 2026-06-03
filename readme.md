This script lets you set a custom GPU fan curve on a headless Linux server. It works without an X server or display — no Xorg, no Wayland, no `nvidia-settings` required.

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 595.71       Driver Version: 595.71       CUDA Version: 13.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 3090      Off  | 00000000:06:00.0 Off |                  N/A |
| 40%   38C    P2   210W / 350W |  18220MiB / 24576MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX 3090      Off  | 00000000:21:00.0 Off |                  N/A |
| 52%   55C    P2   280W / 350W |  18216MiB / 24576MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
```

The tool uses NVML (via `pynvml`) to talk directly to the NVIDIA driver. It automatically discovers which fans belong to which GPU, so multi-fan GPUs and multi-GPU systems are handled correctly.

### Requirements

- NVIDIA driver ≥ 465 (for `nvmlDeviceSetFanSpeed_v2`)
- Must run as root
- `nvidia-persistenced` recommended for headless systems

### Installation

```bash
pip install coolgpus
```

For use as a systemd service, install system-wide so the binary is in `/usr/local/bin`:

```bash
sudo pipx install coolgpus --global
# or: sudo pip install coolgpus --break-system-packages
```

If you use Anaconda and prefer to avoid system Python, point the service file at your Anaconda binary instead (see systemd section below).

### Usage

```bash
sudo $(which coolgpus) --speed 99 99
```
If you hear your server take off, it works! Now interrupt it and re-run with sensible defaults:
```bash
sudo $(which coolgpus)
```
Or pass your own fan curve:
```bash
sudo $(which coolgpus) --temp 17 84 --speed 15 99
```
This makes fan speed increase linearly from 15% at <17C to 99% at >84C. You can increase `--hyst` to smooth out oscillations at the cost of fans possibly running slightly faster than needed.

Control how often the script checks temps and adjusts fans with `--interval`:
```bash
sudo $(which coolgpus) --interval 5
```
The default is 7 seconds.

#### Piecewise Linear Control
You can list any sequence of (increasing!) temperatures and speeds, linearly interpolated:
```bash
sudo $(which coolgpus) --temp 20 55 80 --speed 5 30 99
```
Fan speed will be 5% at <20C, increasing linearly to 30% at 55C, then to 99% at 80C.

#### Fan Start Threshold

Modern GPU fans have a physical minimum speed below which they won't start from rest (the "fan stop" or "0dB" feature common on RTX 3000/4000 series cards). The default minimum speed is 40%, which reliably starts fans on RTX 3090s. If your fans aren't spinning up, try raising the floor:

```bash
sudo $(which coolgpus) --speed 50 99
```

#### Thermal Protection

If a GPU exceeds `--max-temp` (default 75C), coolgpus will reduce its power limit in `--power-step` watt steps (default 25W) down to a floor of 33% of the default TDP. The power limit is never raised — coolgpus only throttles down.

#### systemd

Copy the service file and reload:

```bash
sudo cp coolgpus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now coolgpus
```

The shipped service file:
```ini
[Unit]
Description=Headless GPU Fan Control
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
ExecStart=/usr/local/bin/coolgpus
Restart=on-failure
RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

If `coolgpus` isn't in `/usr/local/bin` (e.g. you're using Anaconda), edit `ExecStart` to use the full path:
```bash
sudo sed -i 's|ExecStart=.*|ExecStart=/home/brian/anaconda3/bin/coolgpus|' /etc/systemd/system/coolgpus.service
sudo systemctl daemon-reload && sudo systemctl restart coolgpus
```

### Driver Compatibility

| Driver | Status | Notes |
|--------|--------|-------|
| ≥ 525 | ✅ Recommended | Full NVML fan control support |
| 465–524 | ✅ Should work | `nvmlDeviceSetFanSpeed_v2` available; `nvmlDeviceGetMinMaxFanSpeed` not available (falls back to 0–100% range) |
| 570.x | ⚠️ Known issues | Fan speed reporting via `nvidia-smi` unreliable under NVML manual control; fans may appear as 0% even when spinning |
| < 465 | ❌ Not supported | `nvmlDeviceSetFanSpeed_v2` not available |

Check your driver version with `nvidia-smi`.

### Troubleshooting

- **`coolgpus: command not found`**: the pip script folder isn't on your PATH. Find it with `pip show coolgpus` or use the full path `sudo /path/to/coolgpus`.
- **`NVMLError_NoPermission`**: must run as root (`sudo`).
- **`NVMLError_DriverNotLoaded`**: NVIDIA driver isn't loaded. Check `nvidia-smi` works first.
- **Fans not spinning despite coolgpus reporting a speed**: your card may have a fan-stop feature that prevents spinning below a certain PWM. Try raising `--speed` floor to 50% or higher.
- **`nvidia-smi` shows 0% fan but GPU is cooling**: known reporting bug on some driver versions (notably 570.x) under NVML manual control. Watch temperature instead of reported fan speed to verify operation.
- **One GPU's fans not responding**: test with `python3 -c "import pynvml; pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(N); pynvml.nvmlDeviceSetFanSpeed_v2(h, 0, 80); pynvml.nvmlShutdown()"` (replace N with GPU index). If no error but fan doesn't move, it may be a hardware fault on that fan header.
- **Service has no journal output**: add `Environment=PYTHONUNBUFFERED=1` to the `[Service]` section of the service file.

### How it works

`coolgpus` uses NVML (NVIDIA Management Library) via the `pynvml` Python bindings to communicate directly with the NVIDIA driver — no X server or display required. On startup it discovers all GPUs and their fans, then loops every few seconds to read temperatures and set fan speeds according to your curve. When the script exits, it returns fan control to the driver.

### Credit

This is a fork maintained by [tazitoo](https://github.com/tazitoo), originally created by [Andy Jones](https://github.com/andyljones/coolgpus). Includes contributions from [kyawakyawa's fork](https://github.com/kyawakyawa/coolgpus).

Earlier ancestry:
* Based on [this 2016 script](https://github.com/boris-dimitrov/set_gpu_fans_public) by [Boris Dimitrov](dimiroll@gmail.com), which is in turn based on [this 2011 script](https://sites.google.com/site/akohlmey/random-hacks/nvidia-gpu-coolness) by [Axel Kohlmeyer](akohlmey@gmail.com).
* [Vladimir Iashin](https://github.com/v-iashin) added piecewise linear control
* [Xun Chen](https://github.com/morfast) fixed the systemctl stop response
