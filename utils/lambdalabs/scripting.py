import subprocess
import json
import socket
import os

def execute_shell_command(command):
    try:
        # Execute the shell command
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return None

def process_json_output(output):
    try:
        # Parse the output as JSON
        json_data = json.loads(output)
        return json_data
    except json.JSONDecodeError:
        print("The output is not valid JSON.")
        return None

# extract contents of .pub file as a string
def get_ssh_key():
    return os.environ.get("LAMBDA_SSH_KEY")

def get_api_key():
    api_key = os.environ.get("LAMBDA_API_KEY")

    return api_key

def get_vm_id(api_key=None):

    if api_key is None:
        api_key = get_api_key()

    # get current IP address
    ip = socket.gethostname().replace("-", ".")

    # search for IP address in current running Lambda Labs instances
    find_instances_command = f"curl -u {api_key}: https://cloud.lambdalabs.com/api/v1/instances"
    # import pdb; pdb.set_trace() # debug
    instances_output = execute_shell_command(find_instances_command)
    all_vms_json = process_json_output(instances_output)
    found_ip = False
    for vm in all_vms_json['data']:
        if vm['ip'] == ip:
            vm_id = vm['id']
            found_ip = True
            return vm_id
    
    if not found_ip:
        print("LambdaLabs IP not found")
        return
    
def terminate_lambdalabs_instance():

    api_key = get_api_key()
    vm_id = get_vm_id(api_key)

    if vm_id is None:
        print("No Lambda Labs instance found with the current IP address, no instance will be terminated.")
        return

    # terminate VM if IP address found
    terminate_command = f"curl -u {api_key}: https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -d '{{\"instance_ids\":[\"{vm_id}\"]}}'"
    terminate_output = execute_shell_command(terminate_command)
    print(terminate_output)
    return
    
def launch_lambdalabs_instance(region_to_file_system_dict, instance_type="gpu_1x_a100_sxm4"):
    api_key = get_api_key()
    
    selected_region = None
    for region in region_to_file_system_dict:
        if is_available(region, instance_type, api_key):
            print(f"Instance type {instance_type} is available in region {region}.")
            selected_region = region
            break
    
    if selected_region is None:
        raise ValueError(f"Instance type {instance_type} is not available in any given region.")

    selected_file_system = region_to_file_system_dict[selected_region]
    print(f"Selected file system: {selected_file_system}")
    
    launch_request = {
        "region_name": selected_region,
        "instance_type_name": instance_type,
        "ssh_key_names": [
            os.environ["LAMBDA_SSH_KEY_NAME"]
        ],
        "file_system_names": [
            selected_file_system
        ],
        "name": f"{region}_{instance_type}",
    }

    launch_command = f"curl -u {api_key}: https://cloud.lambda.ai/api/v1/instance-operations/launch -X POST -d '{json.dumps(launch_request)}'"
    launch_output = execute_shell_command(launch_command)
    launch_output_dict = process_json_output(launch_output)

    if not launch_output_dict:
        raise ValueError("Error launching instance: No output from Lambda Labs API")
    elif "error" in launch_output_dict:
        raise ValueError(f"Error launching instance: {launch_output_dict['error']}")

    if "data" in launch_output_dict and "instance_ids" in launch_output_dict["data"]:
        instance_ids = launch_output_dict["data"]["instance_ids"]
        print(f"Launched instance with ID {instance_ids[0]}")


def get_available_regions(instance_type, allowed_regions=None, api_key=None):
    """Names of regions with capacity for instance_type (optionally restricted to allowed_regions)."""
    instance_types_dict = get_available_lambdalabs_instances(api_key)
    if instance_types_dict is None or instance_type not in instance_types_dict["data"]:
        return []

    regions = [r["name"] for r in instance_types_dict["data"][instance_type]["regions_with_capacity_available"]]
    if allowed_regions is not None:
        regions = [r for r in regions if r in allowed_regions]
    return regions

def get_file_systems_in_region(region, api_key=None):
    if api_key is None:
        api_key = get_api_key()

    output = execute_shell_command(f"curl -u {api_key}: https://cloud.lambda.ai/api/v1/file-systems")
    output_dict = process_json_output(output)
    if not output_dict or "data" not in output_dict:
        return []

    return [fs["name"] for fs in output_dict["data"] if fs["region"]["name"] == region]

def launch_instance_in_region(region, instance_type, api_key=None):
    """Launch instance_type in region, attaching any existing file systems in that region. Returns the instance id."""
    if api_key is None:
        api_key = get_api_key()

    launch_request = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [os.environ["LAMBDA_SSH_KEY_NAME"]],
        "file_system_names": get_file_systems_in_region(region, api_key),
        "name": f"{region}_{instance_type}",
    }

    launch_command = f"curl -u {api_key}: https://cloud.lambda.ai/api/v1/instance-operations/launch -X POST -d '{json.dumps(launch_request)}'"
    launch_output_dict = process_json_output(execute_shell_command(launch_command))

    if not launch_output_dict:
        raise ValueError("Error launching instance: No output from Lambda Labs API")
    if "error" in launch_output_dict:
        raise ValueError(f"Error launching instance: {launch_output_dict['error']}")

    return launch_output_dict["data"]["instance_ids"][0]


def get_available_lambdalabs_instances(api_key=None):
    if api_key is None:
        api_key = get_api_key()

    find_instances_command = f"curl -u {api_key}: https://cloud.lambda.ai/api/v1/instance-types"
    find_instances_output = execute_shell_command(find_instances_command)
    instance_types_dict = process_json_output(find_instances_output)

    if not instance_types_dict:
        return None

    if "error" in instance_types_dict:
        raise ValueError(f"Error retrieving instance types: {instance_types_dict['error']}")

    return instance_types_dict

def is_available(target_instance_type, allowed_regions=None, api_key=None):

    if api_key is None:
        api_key = get_api_key()
    
    instance_types_dict = get_available_lambdalabs_instances(api_key)
    if instance_types_dict is None:
        return False
    if target_instance_type in instance_types_dict["data"]:
        target_dict = instance_types_dict["data"][target_instance_type]
        if allowed_regions is not None:
            for region in target_dict["regions_with_capacity_available"]:
                if region["name"] in allowed_regions:
                    return True   
        else:
            if len(target_dict["regions_with_capacity_available"]) > 0:
                return True
    
    return False

def get_instance_status(instance_id, api_key=None):
    if api_key is None:
        api_key = get_api_key()

    while True:
        check_instance_command = f"curl -u {api_key}: -X GET https://cloud.lambda.ai/api/v1/instances/{instance_id} \
-H 'accept: application/json'"

        check_instance_output = execute_shell_command(check_instance_command)
        check_instance_dict = process_json_output(check_instance_output)

        if "error" in check_instance_dict:
            print(f"Error checking instance status: {check_instance_dict['error']}")
            return None
        elif "data" in check_instance_dict:
            status = check_instance_dict["data"]["status"]
            return status
        else:
            print(f"Unexpected response format: {check_instance_dict}")
            return None


def get_all_instances(api_key=None):
    """List of instance dicts (id, name, status, ...) for all instances, or None on error."""
    if api_key is None:
        api_key = get_api_key()

    output = execute_shell_command(f"curl -s -u {api_key}: https://cloud.lambda.ai/api/v1/instances -H 'accept: application/json'")
    output_dict = process_json_output(output) if output else None

    if not output_dict or "data" not in output_dict:
        print(f"Error listing instances: {output_dict}")
        return None
    return output_dict["data"]