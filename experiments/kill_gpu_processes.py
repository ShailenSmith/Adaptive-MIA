import os
import signal
import subprocess
import time
import psutil
import re
from typing import List, Dict
from subprocess import Popen, PIPE

# kill only processes using GPU memory that belong to the current user

def get_gpu_processes() -> List[Dict[str, str]]:
    try:
        output = subprocess.check_output("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader", shell=True).decode("utf-8")
        lines = output.strip().split("\n")
        processes = []
        for line in lines:
            parts = line.split(", ")
            if len(parts) == 2:
                pid = int(parts[0])
                process_name = parts[1]
                processes.append({"pid": pid, "process_name": process_name})
        return processes
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        return []

def kill_process(pid: int) -> None:
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        print(f"Killed process {pid}")
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist.")
    except psutil.TimeoutExpired:
        print(f"Process {pid} did not terminate in time, killing it forcefully.")
        process.kill()
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
    if psutil.pid_exists(pid):
        print(f"Process {pid} is still running after termination attempt.")
    else:
        print(f"Process {pid} has been successfully terminated.")

current_uid = os.getuid()

processes = get_gpu_processes()
if not processes:
    print("No GPU processes found.")
else:
    print(f"Found {len(processes)} GPU processes.")
    killed_count = 0
    for process in processes:
        pid = process["pid"]
        process_name = process["process_name"]
        try:
            proc = psutil.Process(pid)
            if proc.uids().real == current_uid:
                print(f"Killing process {pid} ({process_name}) owned by current user.")
                kill_process(pid)
                killed_count += 1
            else:
                print(f"Skipping process {pid} ({process_name}) not owned by current user.")
        except Exception as e:
            print(f"Could not check user for process {pid}: {e}")
    print(f"All GPU processes owned by current user killed. Total killed: {killed_count}")