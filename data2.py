
with open("wonderland.txt", "r") as f:
    book = f.read()

chars = list(set(book)) #set() is an unordered collection with no duplicate elements.

book_size, chars_size = len(book), len(chars)

print ("data has {0} characters, {1} is unique. ".format(book_size, chars_size))

char_to_idx = { ch:idx for idx,ch in enumerate(chars)}
idx_to_char = { idx:ch for idx, ch in enumerate(chars)}
