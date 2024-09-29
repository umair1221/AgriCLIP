import tarfile
import os
import sys

def extract_tar(tar_path, extract_path='.'):
    try:
        # Get the file size
        file_size = os.path.getsize(tar_path)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get list of all members
            members = tar.getmembers()
            total_members = len(members)
            
            # Extract all the contents
            for i, member in enumerate(members, 1):
                tar.extract(member, path=extract_path)
                
                # Calculate progress
                progress = (i / total_members) * 100
                
                # Calculate extracted size
                extracted_size = sum(m.size for m in members[:i])
                
                # Print progress
                print(f'\rProgress: {progress:.2f}% - Extracted: {extracted_size / 1e9:.2f} GB / {file_size / 1e9:.2f} GB', end='')
                sys.stdout.flush()

        print("\nExtraction completed successfully!")
        
    except tarfile.ReadError as e:
        print(f"Error reading the tar file: {e}")
    except PermissionError:
        print("Permission denied. Try running the script with sudo.")
    except OSError as e:
        print(f"OS error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    tar_path = './Output/preData.tar.gz'  # replace with your file path if different
    extract_path = '.'  # current directory, change if needed
    
    extract_tar(tar_path, extract_path)