
def string_to_char_list(string):
    return [x for x in string]

def concat_chars(clist):
    return "".join(clist)

lowercases = concat_chars([chr(i) for i in range(97,123)])
uppercases = concat_chars([chr(i) for i in range(65,91)])
numbers = concat_chars([chr(i) for i in range(48,58)])
punctuation = ".?!,:;\'\"-/()"
breaks = " \n"
good_chars = lowercases + uppercases + punctuation + breaks

def only_keep(chars, string):
    return filter(lambda c: c in chars, string)
