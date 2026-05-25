#!/usr/bin/env python3

"""Write a Python function that returns all students sorted
by average score:

    """


def top_students(mongo_collection):
    """function that returns all students sorted by average score:


    Args:
        mongo_collection (_type_): _description_
    """
    return list(mongo_collection.find().sort("averageScore", -1))
