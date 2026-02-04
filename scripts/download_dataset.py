import kagglehub

# Download latest version
path = kagglehub.dataset_download("rohulaminlabid/iotid20-dataset")

print("Path to dataset files:", path)