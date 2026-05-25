#!/usr/bin/env python3

"""Write a Python script that provides some stats about Nginx
    logs stored in MongoDB:
    """
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client.logs
    collection = db.nginx

    x = collection.count()
    print(f"{x} logs")
    out = []
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods")
    for method in methods:
        count = collection.count({'method': method})
        print(f"    method {method}: {count}")
