import kagglehub

# Download latest version
path = kagglehub.dataset_download("drscarlat/syntheacovid100k")

print("Path to dataset files:", path)
