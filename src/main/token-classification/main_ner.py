from typing import List

import pyarrow as pa
from datasets import ClassLabel
from pyarrow import csv
from pyarrow._csv import ParseOptions


def init_apache_arrow():
    table = csv.read_csv("./data/train.csv", parse_options=ParseOptions(delimiter="\t"))
    # label = table.column("label")
    # # label.
    # # for i in label:
    # indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
    #
    # dictionary = pa.array(['foo', 'bar', 'baz'])
    #
    # dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    # print(dict_array)
    #
    # unique = label.unique()
    # unique_dict = {}
    # for idx, key in enumerate(unique):
    #     unique_dict[key] = idx
    # # table.add_column()
    return table


def dictionary():
    table = csv.read_csv("./data/train.csv", parse_options=ParseOptions(delimiter="\t"))
    # datasets = load_dataset("csv", data_files="./data/train.csv", delimiter="\t", quoting=csv_lib.QUOTE_NONE)
    # train_dataset: Dataset = datasets["train"]
    # train_dataset = Dataset(arrow_table=table)
    # table = train_dataset.data
    aa = set(table.column("label").to_pylist())

    class_label_ = table.column("label").unique()
    class_label = ClassLabel(num_classes=len(class_label_), names=class_label_.tolist())
    # ner_ids_list: ChunkedArray = class_label.str2int(label.column('label').to_numpy())
    return class_label


def process_data(file_name, class_label):
    # class_label: ClassLabel = dictionary()
    sentence_list: List = []
    text = ""
    header = ""
    with open(file_name) as f:
        header = f.readline()
        content = f.readlines()
        for line in content:
            if line != "\n":
                text += line
            else:
                sentence_list.append(text)
                text = ""
            # print(line)
    tokens_list: List = []
    labels_list: List = []
    index_list: List = []
    for index, sentence in enumerate(sentence_list):

        rows = sentence.split('\n')
        tokens = []
        labels = []

        for row in rows:
            row_list = row.split("\t")
            if len(row_list) == 2:
                token = row_list[0]
                label = row_list[1]
                tokens.append(token)
                labels.append(label)


        tokens_list.append(tokens)
        labels_list.append(class_label.str2int(labels))
        index_list.append(index)

    tokens_column = pa.array(tokens_list)
    labels_column = pa.array(labels_list)
    py_table = pa.Table.from_arrays(arrays=[index_list, tokens_column, labels_column], names=["id", "tokens", "ner_tags"])
    return py_table


if __name__ == "__main__":


    # bb = pa.Table.from_batches([tokens_column, labels_column], ["tokens", "ner_tags"))
    # table.append_column(tokens_column)
    # table.append_column(labels_column)
    print("123123")

    # ner_array = pa.array(ner_ids_list)
    # # table.add_column(2, "ner_ids", ner_array)
    #
    # class_label = Sequence(feature=ClassLabel(num_classes=len(class_label_), names=class_label_.tolist()))
    #
    # aaa = train_dataset.data.append_column("ner_tags", ner_array)
    # train_dataset._data = aaa
    print("asdasd")
