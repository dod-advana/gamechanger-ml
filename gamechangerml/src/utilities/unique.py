def unique(seq):
    """Returns the unique items of the list, preserving order.

    Args:
        seq (list)
    Returns:
        list
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
