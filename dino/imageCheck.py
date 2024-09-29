import os
from PIL import Image
import traceback
from tqdm import tqdm

def validate_images(image_directory, output_file):
    bad_files = []
    for subdir, dirs, files in tqdm(os.walk(image_directory)):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # I'm verifying the integrity of the file here
                    # print("Done")
            except Exception as e:
                print(f"Problem with file: {filepath}")
                bad_files.append(filepath)
                print(traceback.format_exc())  # This will print more details about the error

    # Write the bad files to an output file
    with open(output_file, 'w') as f:
        for item in bad_files:
            f.write("%s\n" % item)
    print(f"List of corrupted/truncated images saved to {output_file}")

# Usage
image_directory = '/share/sdb/umairnawaz/Thesis_Work/Data/pre-train-data'
output_file = './output/bad_images_list.txt'
validate_images(image_directory, output_file)
