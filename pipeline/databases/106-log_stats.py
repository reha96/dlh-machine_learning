#!/usr/bin/env python3

"""Improve 34-log_stats.py by adding the top 10
of the most present IPs
in the collection nginx of the database logs:
    """
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client.logs
    collection = db.nginx

    x = collection.count_documents({})
    print(f"{x} logs")
    out = []
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = collection.count_documents({'method': method})
        print(f"\tmethod {method}: {count}")
    status = collection.count_documents({'method': "GET", 'path': "/status"})
    print(f"{status} status check")
    print("IPs:")

    pipeline = [
        {"$group": {
            "_id": "$IP",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    out = collection.aggregate(pipeline)
    for doc in out:
        print(f"\t{doc['_id']}: {doc['count']}")
