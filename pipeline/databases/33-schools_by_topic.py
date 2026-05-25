#!/usr/bin/env python3

"""Write a Python function that returns the list
    of school having a specific topic:
    """


def schools_by_topic(mongo_collection, topic):
    """returns the list
    of school having a specific topic

    Args:
        mongo_collection (_type_): _description_
        topic (_type_): _description_
    """
    return mongo_collection.find({'topic': topic})
