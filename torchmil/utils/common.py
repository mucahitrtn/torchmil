import csv
import os

def read_csv(file):
    """
    Read a CSV file and return a dictionary where the keys are the column names and the values are lists of the corresponding column values.
    """
    dict_list = []
    with open(file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dict_list.append(row)
    return dict_list

def keep_only_existing_files(path, names, ext='.npy'):
    """
    Keep only the files that exist in the given path.    
    """
    existing_files = []
    for name in names:
        file = f'{path}/{name}{ext}'
        if os.path.isfile(file):
            existing_files.append(name)
    return existing_files