def delete(s, i):
    if len(s) > i:
        return ''.join((s[:i], s[i + 1:]))
    else:
        return s


def rules_delete(j):
    sli = [
        delete(j, 0),
        delete(j, 1),
        delete(j, 2),
        delete(j, 3),
        delete(j, 4)
    ]
    unique_list = []
    # unique_list.append('(')
    for x in sli:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
