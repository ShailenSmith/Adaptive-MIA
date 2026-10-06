from scripting import terminate_lambdalabs_instance #! relative import that will not work if the file is moved
import subprocess
import time

def get_gpu_memory(used=True):
    # use nvidia-smi to see which GPUs are available
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv", shell=True).decode("utf-8")
    gpu_lines = gpu_info.strip().split("\n")[1:]  # Skip the header line
    gpu_memory = {}
    for line in gpu_lines:
        parts = line.split(", ")
        gpu_index = int(parts[0])
        memory_used = int(parts[1].split(" ")[0])
        memory_free = int(parts[2].split(" ")[0])
        if used:
            gpu_memory[gpu_index] = memory_used
        else:
            gpu_memory[gpu_index] = memory_free
    return gpu_memory

def wait_for_all_gpus_empty(wait_empty_interval=60):
    # check if GPUs have no vRAM usage
    used_gpu_memory = get_gpu_memory(used=True)
    times_all_empty = 0
    while True:
        all_empty = True
        for gpu_index, memory_free in used_gpu_memory.items():
            if used_gpu_memory[gpu_index] > 100: 
                print(f"GPU {gpu_index} is still in use with {used_gpu_memory[gpu_index]} MB used.")
                print(f"Checking again in {wait_empty_interval} seconds...")
                times_all_empty = 0
                all_empty = False
                break

        if all_empty:
            times_all_empty += 1
        else:
            times_all_empty = 0

        if times_all_empty == 1:
            print(f"All GPUs are empty for the first time. Checking again after {2*wait_empty_interval} seconds...")
            time.sleep(wait_empty_interval)
        elif times_all_empty > 1:
            print("All GPUs are empty again. Exiting...")
            return 

        time.sleep(wait_empty_interval) # check every 10 minutes
        used_gpu_memory = get_gpu_memory(used=True)

if __name__ == "__main__":
    print("Waiting for all GPUs to be empty...")
    wait_for_all_gpus_empty()
    print("All GPUs are empty. Terminating LambdaLabs instance...")
    terminate_lambdalabs_instance()