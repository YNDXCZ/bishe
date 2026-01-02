import os
import zipfile
import urllib.request
import shutil
import sys

# User provided URL
DOWNLOAD_URL = "https://universe.roboflow.com/ds/2B8ipsH8ll?key=IOAs7JChDM"
TEMP_DIR = "data/temp_download"
ZIP_PATH = os.path.join(TEMP_DIR, "dataset.zip")

def download_and_extract():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
        
    print(f"Downloading from {DOWNLOAD_URL}...")
    try:
        # User agent sometimes required
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        print(f"Extraction complete to {TEMP_DIR}")
        
        # Cleanup zip
        os.remove(ZIP_PATH)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
