import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
for fan_id in [0, 1]:
    try:
        pynvml.nvmlDeviceSetFanControlPolicy(handle, fan_id, pynvml.NVML_FAN_POLICY_MANUAL)
        print(f"Fan {fan_id}: policy set OK")
    except Exception as e:
        print(f"Fan {fan_id}: policy failed: {e}")
    try:
        pynvml.nvmlDeviceSetFanSpeed_v2(handle, fan_id, 60)
        print(f"Fan {fan_id}: speed set OK")
    except Exception as e:
        print(f"Fan {fan_id}: speed failed: {e}")
pynvml.nvmlShutdown()
