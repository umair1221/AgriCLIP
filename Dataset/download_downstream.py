import argparse
import os
import gdown

def parse_args():
    parser = argparse.ArgumentParser(description='Download datasets from Google Drive and save them to a specified directory.')
    parser.add_argument('--output-dir', type=str, default='./Dataset/Downstream-Data',
                        help='Directory where datasets will be saved')
    return parser.parse_args()

# Dictionary of dataset names and corresponding Google Drive links
datasets = {
    "Dataset1": "https://drive.google.com/uc?id=your_drive_id_1",
    "Dataset2": "https://drive.google.com/uc?id=your_drive_id_2",
    "Dataset3": "https://drive.google.com/uc?id=your_drive_id_3"
}

def download_datasets(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    for dataset_name, dataset_link in datasets.items():
        # Create a folder for each dataset
        dataset_folder = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        # Define the output path for the downloaded file
        output_file = os.path.join(dataset_folder, f"{dataset_name}.zip")
        
        # Download the dataset using gdown
        print(f"Downloading {dataset_name}...")
        gdown.download(dataset_link, output_file, quiet=False)
        
        print(f"{dataset_name} downloaded and saved in {dataset_folder}")

if __name__ == "__main__":
    args = parse_args()
    download_datasets(args.output_dir)
