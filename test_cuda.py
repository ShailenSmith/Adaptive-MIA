import torch

print("Cuda available: ", torch.cuda.is_available())  # Should print True
print("Torch version: ", torch.version.cuda)        # Should print 12.1
print("Device: ", torch.cuda.get_device_name(0))  # Should display "H100"