import subprocess
import os
import time
import torch

def run_commands(commands, gpu_helper=None, experiment_type=None, experiment_name="TEST", parallel=False, run_number=0):
    for attack_run_name, cmd in commands.items():
        # empty cuda cache
        torch.cuda.empty_cache()

        # get gpu id and command
        try:
            gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            gpu_id = "unknown"

        # run process in foreground and wait to finish
        print(f"---\n\nLaunching on GPU(s): {gpu_id} //// {attack_run_name} //// with command: {cmd}\n\n")
        try:
            proc = subprocess.check_call(cmd, shell=True) # check_call will wait until the command finishes to continue
            print(f"Process /// {attack_run_name} /// completed.\n\n-----")
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Error occurred while running command: {cmd}\nError: {e}")

        # wait for 5 seconds before launching the next command
        time.sleep(5)

    print("All commands have completed.")

    return