import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Load environment variables
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class NetworkDataExtract:
    def __init__(self):
        try:
            if MONGO_DB_URL is None:
                raise ValueError("MONGO_DB_URL is not set in environment variables")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path: str):
        try:
            # ✅ Validate CSV path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found at: {file_path}")

            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)

            records = list(json.loads(data.T.to_json()).values())
            logging.info(f"CSV loaded successfully with {len(records)} records")

            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            if len(records) == 0:
                raise ValueError("No records to insert into MongoDB")

            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = client[database]
            col = db[collection]

            col.insert_many(records)
            logging.info(f"Inserted {len(records)} records into MongoDB")

            return len(records)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    # ✅ OS-independent path (THIS FIXES YOUR ERROR)
    FILE_PATH = os.path.join("Network_Data", "phisingData.csv")

    DATABASE = "Rajveer"
    COLLECTION = "NetworkData"

    networkobj = NetworkDataExtract()

    records = networkobj.csv_to_json_convertor(FILE_PATH)
    print(f"Records extracted: {len(records)}")

    no_of_records = networkobj.insert_data_mongodb(
        records=records,
        database=DATABASE,
        collection=COLLECTION
    )

    print(f"Records inserted into MongoDB: {no_of_records}")
