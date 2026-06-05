"""Scratch test for train_bpe file loading."""

input_path = "data/tinystories.txt"

with open(input_path, "r", encoding="utf-8") as file:
    content = file.read()

print("Total characters:", len(content))
print("First 200 chars:", repr(content[:200]))
print("First 200 as byte ints:", list(content[:20].encode("utf-8")))
