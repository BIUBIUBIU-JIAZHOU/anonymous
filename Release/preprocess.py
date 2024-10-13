import math
import os
import random
import sys

sys.path.append(".")

join = os.path.join


# def parse_tuple(sentence, lists, task):
#     words = sentence.split()
#     if task in ['acos', 'asqp']:
#         for quadruple_index, quadruple in enumerate(lists):
#             A = quadruple[0]
#             O = quadruple[3]
#             if A != 'null':
#                 A_list = find_word_indices(words, A)
#                 lists[quadruple_index][0] = A_list
#             if O != 'null':
#                 # print(f"words: {words}")
#                 # print(f"O: {O}")
#                 O_list = find_word_indices(words, O)
#                 lists[quadruple_index][3] = O_list
#     if task in ['tasd']:
#         lists = [list(tup) for tup in lists]
#         for triple_index, triple in enumerate(lists):
#             A = triple[0]
#             if A != 'null':
#                 # print(f"words: {words}")
#                 # print(f"A: {A}")
#                 A_list = find_word_indices(words, A)
#                 lists[triple_index][0] = A_list
#     return lists


# def find_word_indices(sent, element):
#     element = element.split()
#     element_list = []
#     for item in element:
#         element_list.append(sent.index(item))
#     return element_list


def process(data_folder, task, data_name, out_dir, ratio):
    """
    1. Aggregate all train, dev, and test sets for the tasks acos/asqp/aste/tasd.
    2. Remove data contamination: delete the test set data that exists in the train/dev sets.
    3. Output data.txt
    Data format: (task, data, words, tuples)
    """
    train_data = []
    dev_data = []
    test_data = []

    # merge all data
    task_path = join(data_folder, task)
    print("task:", task_path)
    data_path = join(task_path, data_name)
    print("data:", data_path)
    # acos data_name
    for split in ["train", "dev", "test"]:
        with open(join(data_path, "{}.txt".format(split)),
                  'r',
                  encoding="utf-8") as fp:
            for line in fp:
                line = line.strip().lower()
                if line != '':
                    words, tuples = line.split('####')
                if split == "test":
                    test_data.append((words, tuples))
                elif split == "train":
                    train_data.append((words, tuples))
                else:
                    dev_data.append((words, tuples))
    # print(train_data)

    for seed in [3407]:
        os.makedirs(out_dir + "/seed{}".format(seed), exist_ok=True)
        random.seed(seed)
        random.shuffle(train_data)
        # idx = int(len(train_data_dedup) * 0.9)
        # train_set = train_data_dedup[:idx]
        # dev_set = train_data_dedup[idx:]

        # sort
        # train_data = sorted(train_data, key=lambda x: x[2])
        # dev_data = sorted(dev_data, key=lambda x: x[2])
        # test_data = sorted(test_data, key=lambda x: x[2])

        with open(out_dir + "/seed{}/train.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            train_len_split = math.ceil(len(train_data) * ratio)
            print('train_len_split: ', train_len_split)
            print(train_data[:train_len_split])
            # train_len_split = 30
            for item in train_data[:train_len_split]:
                fp.write("{}####{}\n".format(*item))

        with open(out_dir + "/seed{}/dev.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            for item in dev_data:
                fp.write("{}####{}\n".format(*item))

        with open(out_dir + "/seed{}/test.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            for item in test_data:
                fp.write("{}####{}\n".format(*item))


def main():
    tasks = ["acos", "asqp"]
    for task in tasks:
        if task == "acos":
            data_names = ['laptop16', 'rest16']
        else:
            data_names = ['rest15', 'rest16']
        for data_name in data_names:
            for ratio in [0.01, 0.05, 0.1]:
                process("data", task, data_name, f"data/ratio/{task}/{data_name}-ratio/{ratio}", ratio)


if __name__ == "__main__":
    main()
