def rules_symphonious(j):
    """ rules for symphonious """
    s = [
        j.replace("ه", "ح"),
        j.replace("ح", "ه"),
        j.replace("ت", "ط"),
        j.replace("ط", "ت"),
        j.replace("ع", "ا"),
        j.replace("ا", "ع"),
        j.replace("غ", "ق"),
        j.replace("ق", "غ"),
        j.replace("ث", "س"),
        j.replace("ص", "س"),
        j.replace("س", "ص"),
        j.replace("س", "ث"),
        j.replace("ض", "ز"),
        j.replace("ظ", "ز"),
        j.replace("ذ", "ز"),
        j.replace("ز", "ض"),
        j.replace("ظ", "ض"),
        j.replace("ذ", "ض"),
        j.replace("ز", "ظ"),
        j.replace("ذ", "ظ"),
        j.replace("ض", "ظ"),
        j.replace("ز", "ذ"),
        j.replace("ض", "ذ"),
        j.replace("ظ", "ذ")
    ]
    unique_list = []
    # unique_list.append('(')
    for x in s:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
