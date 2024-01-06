def insert_character(s, i):
    if len(s) > i:
        switcher = {
            "ط": "ظ",
            "ز": "ط",
            "ر": "ز",
            "ذ": "ر",
            "د": "ذ",
            "و": "د",
            "ض": "ص",
            "ص": "ض",
            "ث": "ص",
            "ق": "ث",
            "ف": "ق",
            "غ": "ف",
            "ع": "غ",
            "ه": "ع",
            "خ": "ه",
            "ح": "خ",
            "ج": "ح",
            "چ": "ج",
            "پ": "چ",
            "ش": "س",
            "س": "ش",
            "ی": "س",
            "ب": "ی",
            "ل": "ب",
            "ا": "ل",
            "ت": "ا",
            "ن": "ت",
            "گ": "ک",
            "ک": "گ"
        }
        s2 = switcher.get(s[i], "")
        return ''.join((s[:i], s2, s[i], s[i + 1:]))
    else:
        return s


def insert_character1(s, i):
    if len(s) > i:
        switcher = {
            "ض": "ص",
            "ص": "ث",
            "ث": "ق",
            "ق": "ف",
            "ف": "غ",
            "غ": "ع",
            "ع": "ه",
            "ه": "خ",
            "خ": "ح",
            "ح": "ج",
            "ج": "چ",
            "چ": "پ",
            "پ": "چ",
            "ظ": "ط",
            "ط": "ز",
            "ز": "ر",
            "ر": "ذ",
            "ذ": "د",
            "د": "ئ",
            "ئ": "و",
            "و": "ئ",
            "ش": "س",
            "س": "ی",
            "ی": "ب",
            "ب": "ل",
            "ل": "ا",
            "ا": "ت",
            "ت": "ن",
            "ن": "م",
            "م": "ک",
            "گ": "ک",
            "ک": "م"
        }
        s2 = switcher.get(s[i], "")
        return ''.join((s[:i], s2, s[i], s[i+1:]))
    else:
        return s


def rules_insert_character(j):
    sli = [
        insert_character(j, 0),
        insert_character(j, 1),
        insert_character(j, 2),
        insert_character(j, 3),
        insert_character(j, 4),
        insert_character1(j, 0),
        insert_character1(j, 1),
        insert_character1(j, 2),
        insert_character1(j, 3),
        insert_character1(j, 4)
    ]
    unique_list = []
    # unique_list.append('(')
    for x in sli:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
