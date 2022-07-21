from json import load


def load_json(path):
    """Load a JSON file.

    Args:
        path (str): Path to the JSON file to load.

    Returns:
        dict
    """
    with open(path) as f:
        file = load(f)
    
    return file
