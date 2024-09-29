import os
import gdown

# Dictionary of dataset names and corresponding Google Drive links
datasets = {
    "Dataset1": "https://drive.google.com/uc?id=your_drive_id_1",
    "Dataset2": "https://drive.google.com/uc?id=your_drive_id_2",
    "Dataset3": "https://drive.google.com/uc?id=your_drive_id_3"
}

# Folder to store all datasets
data_dir = "datasets"
os.makedirs(data_dir, exist_ok=True)

def download_datasets():
    for dataset_name, dataset_link in datasets.items():
        # Create a folder for each dataset
        dataset_folder = os.path.join(data_dir, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Define the output path for the downloaded file
        output_file = os.path.join(dataset_folder, f"{dataset_name}.zip")
        
        # Download the dataset using gdown
        print(f"Downloading {dataset_name}...")
        gdown.download(dataset_link, output_file, quiet=False)
        
        print(f"{dataset_name} downloaded and saved in {dataset_folder}")

if __name__ == "__main__":
    download_datasets()
