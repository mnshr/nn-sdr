import urllib.request
import zipfile
import gzip
import os
import io

# Locate the data directory relative to this script file
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to download with User-Agent to avoid 403 Forbidden errors
def download_url(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urllib.request.urlopen(req) as response:
        with open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
    print("Done.")

def download_and_extract_zip(url, extract_dir, zip_name):
    zip_path = os.path.join(DATA_DIR, zip_name)
    target_extract_dir = os.path.join(DATA_DIR, extract_dir)
    os.makedirs(target_extract_dir, exist_ok=True)
    try:
        download_url(url, zip_path)
        print(f"Extracting {zip_path} to {target_extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_extract_dir)
        print("Extraction complete.")
        os.remove(zip_path) # Clean up the zip file
    except Exception as e:
        print(f"Failed to download/extract {url}: {e}")

# List of datasets to download
# 1. Boston Housing
download_url(
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
    os.path.join(DATA_DIR, "boston_housing.csv")
)

# 2. KC House Data
download_url(
    "https://raw.githubusercontent.com/remijul/dataset/master/kc_house_data.csv",
    os.path.join(DATA_DIR, "kc_house_data.csv")
)

# 3. Beijing Air Quality (UCI ID 501)
download_and_extract_zip(
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
    "beijing_air_quality",
    "beijing_air_quality.zip"
)

# 4. Superconductivity (UCI ID 464)
download_and_extract_zip(
    "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "superconductivity",
    "superconductivity.zip"
)

# 5. Appliances Energy Prediction (UCI ID 374)
download_and_extract_zip(
    "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
    "appliances_energy",
    "appliances_energy.zip"
)

# 6. Seoul Weather (UCI ID 514)
download_and_extract_zip(
    "https://archive.ics.uci.edu/static/public/514/bias+correction+of+numerical+prediction+model+temperature+forecast.zip",
    "seoul_weather",
    "seoul_weather.zip"
)

# 7. CT Slices (UCI ID 206)
download_and_extract_zip(
    "https://archive.ics.uci.edu/static/public/206/relative+location+of+ct+slices+on+axial+axis.zip",
    "ct_slices",
    "ct_slices.zip"
)

# 8. Jena Climate
download_and_extract_zip(
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    "jena_climate",
    "jena_climate.zip"
)

# 9. MNIST
mnist_dir = os.path.join(DATA_DIR, "mnist")
os.makedirs(mnist_dir, exist_ok=True)
mnist_urls = {
    "train_img": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_lbl": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_img": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_lbl": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
}
for name, url in mnist_urls.items():
    download_url(url, os.path.join(mnist_dir, f"{name}.gz"))

# 10. Fashion-MNIST
fmnist_dir = os.path.join(DATA_DIR, "fashion_mnist")
os.makedirs(fmnist_dir, exist_ok=True)
fmnist_urls = {
    "train_img": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_lbl": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_img": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_lbl": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
}
for name, url in fmnist_urls.items():
    download_url(url, os.path.join(fmnist_dir, f"{name}.gz"))
