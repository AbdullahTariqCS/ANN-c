import kagglehub

# Download latest version
path = kagglehub.dataset_download("lopalp/alphanum")

print("Path to dataset files:", path)