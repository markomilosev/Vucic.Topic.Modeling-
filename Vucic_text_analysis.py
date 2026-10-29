import os
import glob

folder_path = ["B:\\Articles\\Vucic\\Posle_nadstresnice", "B:\\Articles\\Vucic\\Pre_nadstresnice"]
arfa1a5 = ["B:\\Articles\\Vucic\\Posle_nadstresnice\\merged1.txt", "B:\\Articles\\Vucic\\Pre_nadstresnice\\merged2.txt"]

# Merging .txt files in each folder into a single file

for path, outfile_name in zip(folder_path, arfa1a5):
    # Get all .txt files in this specific folder
    txt_files = glob.glob(os.path.join(path, "*.txt"))
    txt_files.sort()  # optional: sorts alphabetically for consistent order

    if not txt_files:
        print(f"No .txt files found in {path}.")
    else:
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            for filename in txt_files:
                with open(filename, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")  # adds a newline between files' content

        print(f"Merged {len(txt_files)} files into '{outfile_name}'")

# Counting tokens in the merged files

import tiktoken
import sys

encoding_name = "cl100k_base"


encoding = tiktoken.get_encoding(encoding_name)
for i in arfa1a5:
    with open(i, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = encoding.encode(text)
    count = len(tokens)

    print(f"Token count: {count}")


import re
from collections import Counter



for folder in folder_path:
    with_source = []
    without_source = []
    counter = Counter()

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        # skip the output files if the script was run before
        if filename in ("with_izvor.txt", "without_izvor.txt"):
            continue

        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read()

        matches = re.findall(r"Извор:\s*(.+)", text)

        if matches:
            with_source.append(text)
            for m in matches:
                counter[m.strip()] += 1
        else:
            without_source.append(text)

    with open(os.path.join(folder, "with_izvor.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(with_source))

    with open(os.path.join(folder, "without_izvor.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(without_source))

    print("Folder:", folder)
    print("Files with Извор:", len(with_source))
    print("Files without Извор:", len(without_source))
    for source, count in counter.most_common():
        print(source, "=", count)
    print()