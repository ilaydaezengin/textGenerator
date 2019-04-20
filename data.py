
with open('omerHayyam.txt', 'r') as data:
    book = data.read()

chars = list(set(book))
book_size, vocab_size = len(book), len(chars)

print ('data has %d chars, %d unique' % (book_size, vocab_size))


char_to_idx = { ch:idx for idx,ch in enumerate(chars)}
idx_to_char = { idx:ch for idx, ch in enumerate(chars)}
