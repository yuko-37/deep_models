import urllib.request
import zipfile
import os
import shutil
from PIL import Image


KAGGLE_CATS_DOGS_URL = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"
DOWNLOADED_DIR = "downloaded"
UNZIPPED_DIR = f"{DOWNLOADED_DIR}/PetImages"
DATASETS_DIR = "datasets"


def download_data():
    os.makedirs(DOWNLOADED_DIR, exist_ok=True)
    download_path = f"{DOWNLOADED_DIR}/catsdogs.zip"

    if not os.path.exists(UNZIPPED_DIR):
        if not os.path.exists(download_path):
            urllib.request.urlretrieve(KAGGLE_CATS_DOGS_URL, download_path)
            print(f"Downloaded to {download_path}")
        else:
            print(f"Found data {download_path}")

        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOADED_DIR)
        print(f"Unzipped to {DOWNLOADED_DIR}")
    else:
        print(f"Found unzipped {UNZIPPED_DIR}")


def extract_data_from(entries, folder):
    for file_name in entries:
        file_path = f"{UNZIPPED_DIR}/{folder}/{file_name}"
        if os.path.getsize(file_path) > 0 and file_name.lower().endswith(".jpg"):
            try:
                with Image.open(file_path) as img:
                    if img.mode in ('RGB', 'RGBA'):
                        shutil.copyfile(file_path, f"datasets/{folder}/{file_name}")
            except Exception as e:
                print(f"Skip file {file_name}", e)

def extract_datasets():
    os.mkdir("datasets")
    os.makedirs("datasets/Cat")
    os.mkdir("datasets/Dog")
    cat_images = os.listdir(f"{UNZIPPED_DIR}/Cat")
    dog_images = os.listdir(f"{UNZIPPED_DIR}/Dog")
    extract_data_from(cat_images, "Cat")
    extract_data_from(dog_images, "Dog")
    cat_ds = os.listdir(f"{DATASETS_DIR}/Cat")
    dog_ds = os.listdir(f"{DATASETS_DIR}/Dog")
    print(f"\nCats: {len(cat_ds)} \ {len(cat_images)}")
    print(f"Dogs: {len(dog_ds)} \ {len(dog_images)}")
    print(f"Total: {len(cat_ds) + len(dog_ds)}")


def clean_datasets():
    datasets_path = "datasets"
    if os.path.exists(datasets_path):
        try:
            shutil.rmtree("datasets")
            print(f"Cleaned datasets folder")
        except Exception as e:
            print(f"Error deleting datasets: {e}")
    else:
        print("No datasets directory.")


if __name__ == '__main__':
    clean_datasets()
    download_data()
    extract_datasets()