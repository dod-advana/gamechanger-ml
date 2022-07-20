"""Helper functions for verifying object attributes"""

def verify_attribute_exists(test_obj, obj, attr_name):
    """"Helper function used to verify that an object attribute exists.

    Args:
        test_obj (unittest.TestCase)
        obj (any): The object to verify an attribute of.
        attr_name (str): Name of the attribute to verify,
    """
    test_obj.assertTrue(
        hasattr(obj, attr_name),
        f"`{obj.__class__.__name__}` is missing attribute `{attr_name}`."
    )

def verify_attribute_type(test_obj, obj, attr_name, expected_type):
    """"Helper function used to verify that an object attribute has the correct 
    type.

    Args:
        test_obj (unittest.TestCase)
        obj (any): The object to verify an attribute of.
        attr_name (str): Name of the attribute to verify.
        expected_type (any): The expected type of the attribute.
    """
    attr = getattr(obj, attr_name)
    test_obj.assertIsInstance(
        attr,
        expected_type,
        f"`{obj.__class__.__name__}` attribute `{attr_name}` has incorrect type."
    )
