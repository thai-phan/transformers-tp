from pyarrow import csv
import pyarrow as pa
from pyarrow._csv import ParseOptions

if __name__ == "__main__":
    table = csv.read_csv("./data/train.csv", parse_options=ParseOptions(delimiter="\t"))
    label = table.column("label")
    # label.
    # for i in label:
    indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])

    dictionary = pa.array(['foo', 'bar', 'baz'])

    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    print(dict_array)

    unique = label.unique()
    unique_dict = {}
    for idx, key in enumerate(unique):
        unique_dict[key] = idx
    # table.add_column()

    print("asdasd")
