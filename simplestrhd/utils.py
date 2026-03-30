def all_not_none(*args):
    return not any(a is None for a in args)
