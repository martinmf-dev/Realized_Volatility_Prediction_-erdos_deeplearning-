#This py file is for importing data from kaggle competition, running it is the same as running the ipynb file with the same name, please read the ipynb for more information. 
#The zipped data imported from kaggle will be deleted after the process. 
#The unzipped data is ignored by git due to competition rule that we can not share raw data. 

import kaggle
kaggle.api.authenticate()
import os
print("Starting to download")
os.system("kaggle competitions download -c optiver-realized-volatility-prediction -p ../raw_data")
print("Downloaded data to ../raw_data folder")
import zipfile 
with zipfile.ZipFile("../raw_data/optiver-realized-volatility-prediction.zip","r") as zip_ref:
    zip_ref.extractall("../raw_data/kaggle_ORVP")
print("Unzipped to ../raw_data/kaggle_ORVP")
os.remove("../raw_data/optiver-realized-volatility-prediction.zip")
print("Removed file ../raw_data/optiver-realized-volatility-prediction.zip")