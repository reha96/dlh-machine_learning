"""Write a Python function that lists all documents in a collection:
    """


def list_all(mongo_collection):
    """function that lists all documents in a collection

    Args:
        mongo_collection (_type_): _description_
    """
    print(mongo_collection.list_collection_names())
