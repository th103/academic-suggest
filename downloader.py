import os
import requests
import gzip
import shutil

url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz"

filename = "cc.es.300.bin.gz"
extracted_filename = "cc.es.300.bin"

print("Downloading...")
response = requests.get(url, stream=True)
with open(filename, 'wb') as file:
    shutil.copyfileobj(response.raw, file)

print("Extracting...")
with gzip.open(filename, 'rb') as f_in:
    with open(extracted_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove(filename)

print("Process finished")
