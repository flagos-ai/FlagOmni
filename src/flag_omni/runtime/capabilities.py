"""Compiler-target checks; a CUDA-compatible torch API is not a vendor ID."""


def is_nvidia_sm90(device):
    if device is None or device.type != "cuda":
        return False
    import torch
    import triton

    # Query the selected device, not whichever accelerator was last current.
    with torch.cuda.device(device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and target.arch == 90
