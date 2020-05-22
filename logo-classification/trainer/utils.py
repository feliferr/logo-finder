import re

def read_lines(file_path):
  names = []
  with open(file_path) as f:
    for line in f:
      names.append(line.replace('\n',''))
  return names

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)',text)]