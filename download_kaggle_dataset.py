import os
import json
import kaggle # Make sure you have 'pip install kaggle'
import sys
import argparse

def setup_kaggle_api(kaggle_json_path):
    """
    Sets up the Kaggle API credentials by copying the kaggle.json file
    to the required location (~/.kaggle/kaggle.json).
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    token_target_path = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json_path):
        print(f"Error: Kaggle API token file not found at '{kaggle_json_path}'")
        print("Please download it from your Kaggle account settings.")
        return False

    try:
        # Create the .kaggle directory if it doesn't exist
        os.makedirs(kaggle_dir, exist_ok=True)
        print(f"Ensured Kaggle directory exists: {kaggle_dir}")

        # Copy the token file
        import shutil
        shutil.copyfile(kaggle_json_path, token_target_path)
        print(f"Copied '{kaggle_json_path}' to '{token_target_path}'")

        # Set permissions (read/write for user only) for security
        os.chmod(token_target_path, 0o600)
        print(f"Set permissions for '{token_target_path}' to 600.")
        return True

    except Exception as e:
        print(f"Error setting up Kaggle API token: {e}")
        return False

def download_dataset(dataset_slug, download_path):
    """
    Downloads and unzips a Kaggle dataset.
    """
    print(f"\nAttempting to download dataset: '{dataset_slug}' to '{download_path}'...")

    try:
        # The kaggle library automatically authenticates using ~/.kaggle/kaggle.json
        kaggle.api.authenticate()
        print("Kaggle API authenticated successfully.")

        # Ensure download path exists
        os.makedirs(download_path, exist_ok=True)

        # Download dataset files
        # force=True overwrites existing files if necessary
        # unzip=True automatically extracts the contents if it's a zip file
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=download_path,
            force=True,
            unzip=True
        )
        print("\nDataset download and extraction complete!")
        print(f"Files are located in: '{download_path}'")
        return True

    except Exception as e:
        print(f"\nError downloading dataset '{dataset_slug}': {e}")
        print("Please check:")
        print("  - Your internet connection.")
        print("  - If the dataset slug is correct.")
        print("  - If your API token is valid and correctly placed.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Kaggle using the API.")
    parser.add_argument("dataset_slug",
                        help="The dataset slug in 'owner_username/dataset_name' format (e.g., 'sgluege/noisy-drone-rf-signal-classification-v2').")
    parser.add_argument("-p", "--path", default=".",
                        help="The directory where the dataset should be downloaded and extracted (default: current directory).")
    parser.add_argument("-t", "--token", default="kaggle.json",
                        help="Path to your downloaded kaggle.json API token file (default: 'kaggle.json' in the script's directory).")

    args = parser.parse_args()

    # 1. Set up the Kaggle token
    if not setup_kaggle_api(args.token):
        sys.exit(1) # Exit if token setup failed

    # 2. Download the dataset
    if not download_dataset(args.dataset_slug, args.path):
        sys.exit(1) # Exit if download failed

    print("\nScript finished.")