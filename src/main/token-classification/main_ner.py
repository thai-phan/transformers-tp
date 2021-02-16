from pyarrow import csv
import pyarrow as pa
from pyarrow._csv import ParseOptions
from datasets import ClassLabel, load_dataset, Dataset
import csv as csv_lib

from pyarrow.lib import ChunkedArray


def init_apache_arrow():
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
    return table

if __name__ == "__main__":
    # table = init_apache_arrow()

    datasets = load_dataset("csv", data_files="./data/train.csv", delimiter="\t", quoting=csv_lib.QUOTE_NONE)
    train_dataset: Dataset = datasets["train"]
    # train_dataset = Dataset(arrow_table=table)
    train_dataset.features['list_label'] = ClassLabel(num_classes=3, names=["O", "B-GENE", "I-GENE"])
    ner_ids_list: ChunkedArray = train_dataset.features['list_label'].str2int(train_dataset.data.column('label').to_numpy())
    ner_array = pa.array(ner_ids_list)
    train_dataset.data.add_column(2, "ner_ids", ner_array)
    print("asdasd")
