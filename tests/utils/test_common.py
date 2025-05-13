import os
import tempfile
import csv
import pytest

from torchmil.utils.common import read_csv, keep_only_existing_files


def test_read_csv():
    with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=['name', 'age'])
        writer.writeheader()
        writer.writerow({'name': 'Alice', 'age': '30'})
        writer.writerow({'name': 'Bob', 'age': '25'})
        tmp_name = tmp.name

    result = read_csv(tmp_name)
    assert isinstance(result, list)
    assert result == [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]

    os.remove(tmp_name)


def test_keep_only_existing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        filenames = ['file1', 'file2', 'file3']
        open(os.path.join(tmpdir, 'file1.npy'), 'a').close()
        open(os.path.join(tmpdir, 'file3.npy'), 'a').close()

        result = keep_only_existing_files(tmpdir, filenames)

        assert result == ['file1', 'file3']