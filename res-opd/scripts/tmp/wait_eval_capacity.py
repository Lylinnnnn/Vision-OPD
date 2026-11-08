#!/usr/bin/env python3
"""Wait for GPU headroom and unused ports without stopping other jobs."""

import argparse
import socket
import subprocess
import time


def free_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    free = {}
    for line in result.stdout.splitlines():
        index, amount = (part.strip() for part in line.split(",", 1))
        free[index] = int(amount)
    return free


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def wait_for_capacity(gpus, ports, min_free_mib, timeout_seconds, poll_seconds=15):
    deadline = time.monotonic() + timeout_seconds
    last_report = 0.0
    while True:
        free = free_gpu_memory()
        shortage = {gpu: free.get(gpu, 0) for gpu in gpus if free.get(gpu, 0) < min_free_mib}
        busy_ports = [port for port in ports if not port_is_free(port)]
        if not shortage and not busy_ports:
            print(f"EVAL_CAPACITY=PASS GPUs={','.join(gpus)} ports={ports}", flush=True)
            return
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(f"Capacity wait expired: GPU free MiB={shortage}; busy ports={busy_ports}")
        if now - last_report >= 60 or not last_report:
            print(f"Waiting for capacity: GPU free MiB={shortage}; busy ports={busy_ports}", flush=True)
            last_report = now
        time.sleep(min(poll_seconds, deadline - now))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-list", required=True)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--port-count", type=int, default=1)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--min-free-mib", type=int, default=36000)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    gpus = [value.strip() for value in args.gpu_list.split(",")]
    if (not gpus or len(gpus) != len(set(gpus)) or any(not gpu.isdigit() for gpu in gpus)
            or args.port_count < 1 or args.min_free_mib < 1 or args.wait_seconds < 1
            or not 1024 <= args.base_port <= 65535
            or args.base_port + (args.port_count - 1) * args.port_stride > 65535):
        parser.error("Invalid GPU, port, or wait parameters")
    ports = [args.base_port + index * args.port_stride for index in range(args.port_count)]
    wait_for_capacity(gpus, ports, args.min_free_mib, args.wait_seconds)


if __name__ == "__main__":
    main()
