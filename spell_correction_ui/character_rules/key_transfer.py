def swap(s, i, j):
    if len(s) > i and len(s) > j:
        return ''.join((s[:i], s[j], s[i + 1:j], s[i], s[j + 1:]))
    else:
        return s


def rules_keytransfer(j):
    sli = [
        swap(j, 0, 1),
        swap(j, 1, 2),
        swap(j, 2, 3),
        swap(j, 3, 4),
        swap(j, 4, 5)
    ]
    unique_list = []
    # unique_list.append('(')
    for x in sli:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
