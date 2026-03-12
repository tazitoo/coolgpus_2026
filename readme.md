This script lets you set a custom GPU fan curve on a headless Linux server.

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54       Driver Version: 550.54       CUDA Version: 12.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4090      On  | 00000000:08:00.0 Off |                  N/A |
| 75%   60C    P2   310W / 450W |  18220MiB / 24564MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX 4090      On  | 00000000:41:00.0  On |                  N/A |
| 90%   70C    P2   295W / 450W |  18216MiB / 24564MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
```

The tool automatically discovers which fans belong to which GPU, so multi-fan GPUs (like triple-fan cards) are handled correctly.

### Instructions
```
pip install coolgpus
sudo $(which coolgpus) --speed 99 99
```
If you hear your server take off, it works! Now interrupt it and re-run either with Sensible Defaults (TM),
```
sudo $(which coolgpus)
```
or you can pass your own fan curve with
```
sudo $(which coolgpus) --temp 17 84 --speed 15 99
```
This will make the fan speed increase linearly from 15% at <17C to 99% at >84C.  You can also increase `--hyst` if you want to smooth out oscillations, at the cost of the fans possibly going faster than they need to.

You can also control how often the script checks temps and adjusts fans with `--interval`:
```
sudo $(which coolgpus) --interval 5
```
The default is 7 seconds, which is fine for most setups. Lower values make the fan response snappier but spam nvidia-settings harder.

#### Piecewise Linear Control
More generally, you can list any sequence of (increasing!) temperatures and speeds, and they'll be linearly interpolated:
```
sudo $(which coolgpus) --temp 20 55 80 --speed 5 30 99
```
Now the fan speed will be 5% at <20C, then increase linearly to 30% up to 55C, then again linearly to 99% up to 80C.

#### systemd
If your system uses systemd and you want to run this as a service, there's a `coolgpus.service` file shipped with the repo. Copy it into place and tweak as needed:

```
sudo cp coolgpus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable coolgpus
sudo systemctl start coolgpus
```

The shipped service file looks like this:
```
[Unit]
Description=Headless GPU Fan Control
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
ExecStart=/usr/local/bin/coolgpus --kill
Restart=on-failure
RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```
You'll probably want to edit the `ExecStart` line to point to your install location (find it with `which coolgpus`) and add any flags you want (like `--temp` and `--speed`).

### Driver Compatibility

Not all NVIDIA drivers play nice with fan control. Known problem drivers:

* **560.35.03**: XNVCtrl fan control may produce `BadValue` errors. Fan speed commands silently fail or crash.
* **565.x**: Fans may not appear in `nvidia-settings` at all, so the script can't discover or control them.
* **570.x (open kernel module)**: Fans may get stuck at 30% regardless of what speed you set. This seems specific to the open kernel module variant.

If you're hitting weird issues, check your driver version with `nvidia-smi` and consider up/downgrading. Driver 550.x has been solid in our testing.

### Troubleshooting
* You've got a display attached: it probably won't work, but see [this issue](https://github.com/tazitoo/coolgpus_2026/issues/1) for progress.
* You've got an X server hanging around for some reason: assuming you don't actually need it, run the script with `--kill`, which'll murder any existing X servers and let the script set up its own. Sometimes the OS [might automatically recreate its X servers](https://unix.stackexchange.com/questions/25668/how-to-close-x-server-to-avoid-errors-while-updating-nvidia-driver), and that's [tricky enough to handle that it's up to you to sort out](https://unix.stackexchange.com/questions/25668/how-to-close-x-server-to-avoid-errors-while-updating-nvidia-driver).
* `coolgpus: command not found`: the pip script folder probably isn't on your PATH. On Ubuntu with the apt-get-installed pip, look in `~/.local/bin`.
* You hit Ctrl+C twice and now your fans are stuck at a certain speed: run the script again and interrupt it _once_, then let it shut down gracefully. Double interrupts stop it from handing control back to the driver. Don't double-interrupt things you barbarian.
* General troubleshooting:
    * Read `coolgpus --help`
    * See if `sudo /path/to/coolgpus` actually works
    * Check that `XOrg`, `nvidia-settings` and `nvidia-smi` can all be called from your terminal.
    * Open `coolgpus` in a text editor, add a `import pdb; pdb.set_trace()` somewhere, and [explore till you hit the error](https://docs.python.org/3/library/pdb.html#debugger-commands).

### Why's this necessary?
If you want to install multiple GPUs in a single machine, you have to use blower-style GPUs else the hot exhaust builds up in your case. Blower-style GPUs can get _very loud_, so to avoid annoying customers nvidia artifically limits their fans to ~50% duty. At 50% duty and a heavy workload, blower-style GPUs will hot up to 85C or so and throttle themselves.

Now if you're on Windows nvidia happily lets you override that limit by setting a custom fan curve. If you're on Linux though you need to use `nvidia-settings`, which - as of Sept 2019 - requires a display attached to each GPU you want to set the fan for. This is a pain to set up, as is checking the GPU temp every few seconds and adjusting the fan speed.

This script does all that for you.

### How it works
When you run `coolgpus`, it sets up a single X server with fake displays for your GPUs. It discovers which fans belong to which GPU (important for multi-fan cards), then loops every few seconds to check temperatures and set fan speeds according to your curve. When the script dies, it returns control of the fans to the drivers and cleans up the X server.

### Credit
This is a fork maintained by [tazitoo](https://github.com/tazitoo), originally created by [Andy Jones](https://github.com/andyljones/coolgpus). Includes contributions from [kyawakyawa's fork](https://github.com/kyawakyawa/coolgpus).

Earlier ancestry:
* Based on [this 2016 script](https://github.com/boris-dimitrov/set_gpu_fans_public) by [Boris Dimitrov](dimiroll@gmail.com), which is in turn based on [this 2011 script](https://sites.google.com/site/akohlmey/random-hacks/nvidia-gpu-coolness) by [Axel Kohlmeyer](akohlmey@gmail.com).
* [Vladimir Iashin](https://github.com/v-iashin) added piecewise linear control
* [Xun Chen](https://github.com/morfast) fixed the systemctl stop response
