# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:39:54 2021

@author: serdarhelli
"""

import requests
from zipfile import ZipFile
from io import BytesIO

def download_dataset(save_path):
    r = requests.get("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hxt48yk462-1.zip")
    print("Downloading...")
    z = ZipFile(BytesIO(r.content))    
    z.extractall(save_path)
    print("Completed...")
