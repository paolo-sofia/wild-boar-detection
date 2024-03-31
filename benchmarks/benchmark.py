import argparse
import threading
import time
from collections import Counter

import numpy as np
import psutil
import requests
from tqdm import tqdm


def monitor_process_memory(process_name, event):
    # Find the process ID (PID) of the process
    pid = None
    for proc in psutil.process_iter(["pid", "name"]):
        if pid:
            break

        if process_name not in proc.info["name"]:
            continue

        for parent in proc.parents():
            if "container" not in parent.name():
                continue
            print(f"Monitoring {proc.name()}")
            pid = proc.pid
            break

    if pid is None:
        print(f"Process '{process_name}' not found.")
        return

    memory_usage = []
    while not event.is_set():
        # Get memory usage of the process
        process = psutil.Process(pid)
        memory_usage.append(process.memory_info().rss)
        # Wait for the specified interval
        psutil.cpu_percent(interval=1)

    # Calculate minimum, average, and maximum memory usage
    min_memory = min(memory_usage)
    avg_memory = sum(memory_usage) / len(memory_usage)
    max_memory = max(memory_usage)

    print(f"Minimum Memory Usage: {min_memory / (1024 * 1024):.2f} MB")
    print(f"Average Memory Usage: {avg_memory / (1024 * 1024):.2f} MB")
    print(f"Maximum Memory Usage: {max_memory / (1024 * 1024):.2f} MB")


def send_requests(n: int, url: str, file_path: bytes, event) -> None:
    headers = {
        "Content-Type": "multipart/form-data",
    }

    exec_times: list[float] = []
    response_codes: list[int] = []
    for _ in tqdm(range(n)):
        files = {
            "file": (file_path, open(file_path, "rb")),
        }
        start = time.perf_counter()
        response = requests.post(url, headers=headers, files=files)
        stop = time.perf_counter()
        exec_times.append(stop - start)
        response_codes.append(response.status_code)
        # time.sleep(0.5)
    event.set()
    exec_times: np.ndarray = np.array(exec_times)
    response_codes: Counter = Counter(response_codes)

    print(f"response codes: {response_codes}")
    print(f"exec times min: {exec_times.min():.3f} seconds")
    print(f"exec times max: {exec_times.max():.3f} seconds")
    print(f"exec times avg: {exec_times.mean():.3f} seconds")
    print(f"exec times std: {exec_times.std():.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send POST request with image file")
    parser.add_argument("-n", "--iterations", type=int, required=True, help="Number of iterations")
    parser.add_argument("--url", type=str, required=True, help="URL for the POST request")
    parser.add_argument("--path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--process_name", type=str, required=True, help="Name of the process to monitor")
    args = parser.parse_args()
    print(args)
    event = threading.Event()

    # Start the thread to send requests
    request_thread = threading.Thread(target=send_requests, args=(args.iterations, args.url, args.path, event))
    request_thread.start()

    # Start the thread to measure memory usage
    memory_thread = threading.Thread(target=monitor_process_memory, args=(args.process_name, event))
    memory_thread.start()

    # Wait for both threads to finish
    request_thread.join()
    memory_thread.join()
