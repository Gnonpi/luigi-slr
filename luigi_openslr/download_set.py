# Source:
# http://www.openslr.org/12/
import tarfile

import requests
import shutil

from config import URL_OPENSLR, DEV_CLEAN, DATA_FOLDER


def download_dev_set():
    if DEV_CLEAN.exists():
        return
    with requests.get(URL_OPENSLR, stream=True) as r:
        with open(str(DEV_CLEAN), 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def untar_dev_set():
    tar = tarfile.open(str(DEV_CLEAN))
    tar.extractall(path=str(DATA_FOLDER))
    tar.close()

if __name__=="__main__":
    download_dev_set()
    untar_dev_set()