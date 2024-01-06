def repetition(s, i):
    if len(s) > i:
        return ''.join((s[:i], s[i], s[i], s[i + 1:]))
    else:
        return s


def rules_repetition_character(j):
    sli = [
        repetition(j, 0),
        repetition(j, 1),
        repetition(j, 2),
        repetition(j, 3),
        repetition(j, 4)
    ]
    unique_list = []
    # unique_list.append('(')
    for x in sli:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
