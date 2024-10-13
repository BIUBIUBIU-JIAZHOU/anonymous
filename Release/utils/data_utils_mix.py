from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils.const import *


def get_aspect_labels(labels_in):
    aspect_dict = {
        'NULL': 1,
        'EXPLICIT': 0,
        'BOTH': 2,
    }
    aspect_labels = []
    for ex in labels_in:
        aspects = set([quad[0] for quad in ex])

        if 'null' not in aspects:
            label = aspect_dict['EXPLICIT']
        else:
            if len(aspects) == 1:
                label = aspect_dict['NULL']
            else:
                label = aspect_dict['BOTH']

        aspect_labels.append(label)
    return aspect_labels


def get_opinion_labels(labels_in):
    # print(labels_in)
    opinion_dict = {
        'NULL': 1,
        'EXPLICIT': 0,
        'BOTH': 2,
    }
    opinion_labels = []
    for ex in labels_in:
        opinions = set([quad[3] for quad in ex])

        if 'null' not in opinions:
            label = opinion_dict['EXPLICIT']
        else:
            if len(opinions) == 1:
                label = opinion_dict['NULL']
            else:
                label = opinion_dict['BOTH']

        opinion_labels.append(label)
    return opinion_labels


def collate_func_train(batch):
    data = {}
    batch = [batch[i] for i in range(len(batch))]
    pad_batch_data(batch, data)
    return data


def collate_func_eval(batch):
    data = {}
    pad_batch_data(batch, data)
    return data


def pad_batch_data(cur_batch, cur_data):
    if len(cur_batch) == 0:
        return
    # if k in ['is_label']:

    for k, v in cur_batch[0].items():
        if isinstance(v, torch.Tensor):
            if len(v.shape) == 1:
                cur_data[k] = pad_sequence([x[k].squeeze(0) for x in cur_batch], batch_first=True)
            else:
                rows = [list(map(lambda c: c.squeeze(0), torch.split(x[k], 1, dim=0))) for x in cur_batch]
                cur_data[k] = pad_sequence(list(chain(*rows)), batch_first=True)
        else:
            cur_data[k] = [x[k] for x in cur_batch]


def read_line_examples_from_file(data_path,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, absa_labels, is_labels = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if line != '':
                words, tuples = line.split('####')
                is_label = str(get_aspect_labels([eval(tuples)])[0])
                sents.append(words.split())
                absa_labels.append(eval(tuples))
                is_labels.append(int(is_label))
    if silence:
        print(f"Total examples = {len(sents)}")
        print(is_labels[-1])
    return sents, absa_labels, is_labels


def get_is_label(aspect_label, opinion_label):
    if aspect_label == 0 and opinion_label == 0:
        label = '0'
    elif aspect_label == 1 and opinion_label == 1:
        label = '1'
    else:
        label = '2'
    return label


def read_line_examples_from_file_random(data_path,
                                        lowercase,
                                        silence=True,
                                        task_name='asqp'):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, absa_labels, is_labels = [], [], []
    sents_explicit, sents_implicit, sents_mix = [], [], []
    absa_labels_explicit, absa_labels_implicit, absa_labels_mix = [], [], []
    is_labels_explicit, is_labels_implicit, is_labels_mix = [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if line != '':
                words, tuples = line.split('####')
                if task_name == 'asqp':
                    is_label = str(get_aspect_labels([eval(tuples)])[0])
                else:
                    # print(tuples)
                    aspect_label = get_aspect_labels([eval(tuples)])[0]
                    opinion_label = get_opinion_labels([eval(tuples)])[0]
                    is_label = get_is_label(aspect_label, opinion_label)
                if is_label == '0':
                    sents_explicit.append(words.split())
                    absa_labels_explicit.append(eval(tuples))
                    is_labels_explicit.append(int(is_label))
                elif is_label == '1':
                    sents_implicit.append(words.split())
                    absa_labels_implicit.append(eval(tuples))
                    is_labels_implicit.append(int(is_label))
                else:
                    sents_mix.append(words.split())
                    absa_labels_mix.append(eval(tuples))
                    is_labels_mix.append(int(is_label))
    data_len = min(len(sents_explicit), len(sents_implicit))
    for i in range(data_len):
        sents.append((sents_explicit[i], sents_implicit[i]))
        absa_labels.append((absa_labels_explicit[i], absa_labels_implicit[i]))
        is_labels.append((is_labels_explicit[i], is_labels_implicit[i]))

    if len(sents_explicit) > len(sents_implicit):
        sents_residue = sents_explicit[data_len:]
        absa_labels_residue = absa_labels_explicit[data_len:]
        is_labels_residue = 0
    else:
        sents_residue = sents_implicit[data_len:]
        absa_labels_residue = absa_labels_implicit[data_len:]
        is_labels_residue = 1

    data_len = min(len(sents_residue), len(sents_mix))
    for i in range(data_len):
        sents.append((sents_mix[i], sents_residue[i]))
        absa_labels.append((absa_labels_mix[i], absa_labels_residue[i]))
        is_labels.append((is_labels_mix[i], is_labels_residue))

    if len(sents_residue) < len(sents_mix):
        sents_final = sents_mix[data_len:]
        absa_labels_final = absa_labels_mix[data_len:]
        is_labels_final = 2
    else:
        sents_final = sents_residue[data_len:]
        absa_labels_final = absa_labels_residue[data_len:]
        is_labels_final = is_labels_residue

    for i in range(len(sents_final)):
        sents.append((sents_final[i], None))
        absa_labels.append((absa_labels_final[i], None))
        is_labels.append((is_labels_final, None))
    # for i in range(0, len(sents_residue), 2):
    #     if i == len(sents_residue) - 1:
    #         continue
    #     sents.append((sents_residue[i], sents_residue[i + 1]))
    #     absa_labels.append((absa_labels_residue[i], absa_labels_residue[i + 1]))
    #     is_labels.append((is_labels_residue, is_labels_residue))
    if silence:
        print(f"Total examples = {len(sents)}")
        print(is_labels[-1])
    # zipped_lists = list(zip(sents, absa_labels, is_labels))
    # random.shuffle(zipped_lists)
    # sents, absa_labels, is_labels = zip(*zipped_lists)
    # sents, absa_labels, is_labels = list(sents), list(absa_labels), list(is_labels)
    return sents, absa_labels, is_labels


def read_line_examples_from_file_sentiment(data_path,
                                           lowercase,
                                           silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, absa_labels, is_labels = [], [], []
    all_data = []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        input_positive, input_negative, input_neutral = [], [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if line != '':
                words, tuples, is_label = line.split('####')
                if eval(tuples)[0][2] == 'positive':
                    input_positive.append((words.split(), eval(tuples), int(is_label)))
                elif eval(tuples)[0][2] == 'negative':
                    input_negative.append((words.split(), eval(tuples), int(is_label)))
                else:
                    input_neutral.append((words.split(), eval(tuples), int(is_label)))
        all_data.append(input_positive)
        all_data.append(input_negative)
        all_data.append(input_neutral)

    for data in all_data:
        sents_explicit, sents_implicit = [], []
        absa_labels_explicit, absa_labels_implicit = [], []
        is_labels_explicit, is_labels_implicit = [], []
        for item in data:
            words, tuples, is_label = item
            if is_label == 0:
                sents_explicit.append(words)
                absa_labels_explicit.append(tuples)
                is_labels_explicit.append(is_label)
            else:
                sents_implicit.append(words)
                absa_labels_implicit.append(tuples)
                is_labels_implicit.append(is_label)

        data_len = min(len(sents_explicit), len(sents_implicit))
        for i in range(data_len):
            sents.append((sents_explicit[i], sents_implicit[i]))
            absa_labels.append((absa_labels_explicit[i], absa_labels_implicit[i]))
            is_labels.append((is_labels_explicit[i], is_labels_implicit[i]))

        if len(sents_explicit) > len(sents_implicit):
            sents_residue = sents_explicit[data_len:]
            absa_labels_residue = absa_labels_explicit[data_len:]
            is_labels_residue = 0
        else:
            sents_residue = sents_implicit[data_len:]
            absa_labels_residue = absa_labels_implicit[data_len:]
            is_labels_residue = 1
        for i in range(len(sents_residue)):
            sents.append((sents_residue[i], None))
            absa_labels.append((absa_labels_residue[i], None))
            is_labels.append((is_labels_residue, None))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, absa_labels, is_labels


def read_line_examples_from_file_label(data_path,
                                       lowercase,
                                       silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, absa_labels, is_labels = [], [], []
    all_data = []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        input_label1, input_label2, input_label3 = [], [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if line != '':
                words, tuples, is_label = line.split('####')
                if len(eval(tuples)) == 1:
                    input_label1.append((words.split(), eval(tuples), int(is_label)))
                elif len(eval(tuples)) == 2:
                    input_label2.append((words.split(), eval(tuples), int(is_label)))
                else:
                    input_label3.append((words.split(), eval(tuples), int(is_label)))
        all_data.append(input_label1)
        all_data.append(input_label2)
        all_data.append(input_label3)

    for data in all_data:
        sents_explicit, sents_implicit = [], []
        absa_labels_explicit, absa_labels_implicit = [], []
        is_labels_explicit, is_labels_implicit = [], []
        for item in data:
            words, tuples, is_label = item
            if is_label == 0:
                sents_explicit.append(words)
                absa_labels_explicit.append(tuples)
                is_labels_explicit.append(is_label)
            else:
                sents_implicit.append(words)
                absa_labels_implicit.append(tuples)
                is_labels_implicit.append(is_label)

        data_len = min(len(sents_explicit), len(sents_implicit))
        for i in range(data_len):
            sents.append((sents_explicit[i], sents_implicit[i]))
            absa_labels.append((absa_labels_explicit[i], absa_labels_implicit[i]))
            is_labels.append((is_labels_explicit[i], is_labels_implicit[i]))

        if len(sents_explicit) > len(sents_implicit):
            sents_residue = sents_explicit[data_len:]
            absa_labels_residue = absa_labels_explicit[data_len:]
            is_labels_residue = 0
        else:
            sents_residue = sents_implicit[data_len:]
            absa_labels_residue = absa_labels_implicit[data_len:]
            is_labels_residue = 1
        for i in range(len(sents_residue)):
            sents.append((sents_residue[i], None))
            absa_labels.append((absa_labels_residue[i], None))
            is_labels.append((is_labels_residue, None))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, absa_labels, is_labels


def get_transformed_io(data_path, args):
    if args.combined:
        sents, absa_labels, is_labels = read_line_examples_from_file_random(data_path, lowercase=args.lowercase,
                                                                            task_name=args.task_name)
    else:
        sents, absa_labels, is_labels = read_line_examples_from_file(data_path, lowercase=args.lowercase)

    return sents, absa_labels, is_labels


# 定义自定义的比较函数
def custom_sort(x, priority_order):
    if len(x) < 4:
        return (0, 0)
    # 提取子列表的第一个和第四个元素
    A = x[0]
    O = x[3]
    # print(type(O))
    # 如果第一个元素是字符串"null"，则将其替换为最小值
    if A == "null":
        A_out = float('-inf')
    else:
        A_out = A[-1]
    # 如果第四个元素是字符串"null"，则将其替换为最小值
    # if O == "null":
    if isinstance(O, list):

        O_out = O[-1]
    else:
        O_out = float('-inf')
    # 根据优先次序参数决定排序的优先次序
    if priority_order == "AO":
        return (A_out, O_out)
    elif priority_order == "OA":
        return (O_out, A_out)


def ABSA_format(label):
    all_qua = []
    for qua in label:
        a = qua[0]
        o = qua[3]
        s = qua[2]
        c = qua[1]
        all_qua.append((a, c, s, o))
    label_strs = []
    for qua in all_qua:
        if qua[0] == 'null':
            permutation = ['[A]', '[O]', '[S]', '[C]']
        else:
            permutation = ['[O]', '[C]', '[S]', '[A]']
        mapping = {'[A]': qua[0], '[C]': qua[1], '[S]': qua[2], '[O]': qua[3]}
        label_list = [char + ' ' + mapping[char] for char in permutation]
        label_str = ' '.join(label_list)
        label_strs.append(label_str)

    return " [SSEP] ".join(label_strs)


def ABSA_format_multi(label, permutation):
    all_qua = []
    for qua in label:
        a = qua[0]
        o = qua[3]
        s = qua[2]
        c = qua[1]
        all_qua.append((a, c, s, o))
    label_strs = []
    for qua in all_qua:
        permutation1 = permutation.split(" ")
        mapping = {'[A]': qua[0], '[C]': qua[1], '[S]': qua[2], '[O]': qua[3]}
        label_list = [char + ' ' + mapping[char] for char in permutation1]
        label_str = ' '.join(label_list)
        label_strs.append(label_str)

    return " [SSEP] ".join(label_strs)


def ignore_masked_positions(input_ids, attention_mask):
    # 将 attention_mask 作为布尔掩码应用于 input_ids 和 attention_mask 本身
    mask = attention_mask.bool()
    filtered_input_ids = input_ids[mask]
    filtered_attention_mask = attention_mask[mask]
    return filtered_input_ids, filtered_attention_mask


class ABSADataset(Dataset):
    def __init__(self, tokenizer_t5, data_path, args, data_type):
        super(ABSADataset, self).__init__()
        self.args = args
        self.tokenizer_t5 = tokenizer_t5
        self.task_name = args.task_name
        self.data_name = args.dataset
        self.data_path = data_path
        self.data_type = data_type
        self.topk = args.topk
        self.combined = args.combined
        self.virtual_token = args.virtual_token
        self.implicit_token = args.implicit_token
        self.ctrl_token = args.ctrl_token

        self.inputs_t5 = []
        self.absa_labels = []
        self.is_labels = []
        self.sents = []

        self._build_examples(args.is_pre_seq_len)

    def __getitem__(self, index):
        input_ids_t5 = self.inputs_t5[index]["input_ids"].squeeze()
        input_mask_t5 = self.inputs_t5[index]["attention_mask"].squeeze()
        input_ids_t5, input_mask_t5 = ignore_masked_positions(input_ids_t5, input_mask_t5)

        absa_label = self.absa_labels[index]["input_ids"].squeeze()
        absa_label_mask = self.absa_labels[index]["attention_mask"].squeeze()
        absa_label, absa_label_mask = ignore_masked_positions(absa_label, absa_label_mask)

        return {
            "index": index,
            "input_ids_t5": input_ids_t5,
            "attention_mask_t5": input_mask_t5,
            "absa_labels": absa_label,
            "absa_label_attention_mask": absa_label_mask,
            "is_labels": self.is_labels[index],
            "dataset": self.data_name
        }

    def __len__(self):
        return len(self.inputs_t5)

    def _build_examples(self, is_pre_seq_len):
        if self.args.full_supervise:
            """
            all_inputs：句子
            all_absa_labels：ABSA标签
            all_is_labels：隐式情感分类标签
            """
            all_inputs, all_absa_labels, all_is_labels = get_transformed_io(
                self.data_path, self.args)
            print("Data examples")
            for i in range(3):
                print(all_inputs[i])
                print(all_absa_labels[i])
                print(all_is_labels[i])
            # self.is_labels = all_is_labels
            # print(self.is_labels)

            for i in range(len(all_inputs)):
                if self.combined:
                    self._get_tokenized_combined(all_inputs[i], all_absa_labels[i], all_is_labels[i],
                                                 is_prompt_tokenized)
                else:
                    self._get_tokenized_multi(all_inputs[i], all_absa_labels[i], all_is_labels[i], is_prompt_tokenized)

    def _get_tokenized_combined(self, words, absa_label, is_label, is_prompt_tokenized):
        permutations = optim_orders_all[self.task_name][self.data_name][:self.topk]
        order2token_implicit = {"[A]": "<implicit_vtoken_a>", "[C]": "<implicit_vtoken_c>",
                                "[S]": "<implicit_vtoken_s>", "[O]": "<implicit_vtoken_o>"}
        order2token_explicit = {"[A]": "<explicit_vtoken1>", "[C]": "<explicit_vtoken2>", "[S]": "<explicit_vtoken3>",
                                "[O]": "<explicit_vtoken4>"}
        for permutation in permutations:
            sentiment_elements = permutation.split(" ")
            if absa_label[1] is None:
                prompt_order = ""
                absa_labels_seq = absa_label[0]
                aspect_label = get_aspect_labels([absa_labels_seq])[0]
                opinion_label = get_opinion_labels([absa_labels_seq])[0]
                self.is_labels.append((aspect_label, opinion_label))
                if aspect_label != 0 or opinion_label != 0:
                    if self.virtual_token:
                        for element in sentiment_elements:
                            if element == "[A]" and aspect_label != 0 and self.implicit_token:
                                prompt_order += "[A] <null> "
                            if element == "[O]" and opinion_label != 0 and self.implicit_token:
                                prompt_order += "[O] <null> "
                            else:
                                prompt_order += element + " " + order2token_implicit[element] + " "
                    else:
                        prompt_order = permutation + " "
                    if self.ctrl_token == 'front':
                        input_seq = prompt_order + ' '.join(words[0])
                    else:
                        input_seq = ' '.join(words[0]) + ' ' + prompt_order

                else:
                    if self.virtual_token:
                        for element in sentiment_elements:
                            prompt_order += element + " " + order2token_explicit[element] + " "
                    else:
                        prompt_order = permutation + " "
                    if self.ctrl_token == 'front':
                        input_seq = prompt_order + ' '.join(words[0])
                    else:
                        input_seq = ' '.join(words[0]) + ' ' + prompt_order
            else:
                absa_labels_seq = absa_label[0] + absa_label[1]
                aspect_label = get_aspect_labels([absa_labels_seq])[0]
                opinion_label = get_opinion_labels([absa_labels_seq])[0]
                self.is_labels.append((aspect_label, opinion_label))

                prompt_order = ""
                aspect_label_front = get_aspect_labels([absa_label[0]])[0]
                opinion_label_front = get_opinion_labels([absa_label[0]])[0]
                if aspect_label_front != 0 or opinion_label_front != 0:
                    if self.virtual_token:
                        for element in sentiment_elements:
                            if element == "[A]" and aspect_label_front != 0 and self.implicit_token:
                                prompt_order += "[A] <null> "
                            if element == "[O]" and opinion_label_front != 0 and self.implicit_token:
                                prompt_order += "[O] <null> "
                            else:
                                prompt_order += element + " " + order2token_implicit[element] + " "
                    else:
                        prompt_order = permutation + " "
                    if self.ctrl_token == 'front':
                        input_seq_front = prompt_order + ' '.join(words[0])
                    else:
                        input_seq_front = ' '.join(words[0]) + ' ' + prompt_order
                else:
                    if self.virtual_token:
                        for element in sentiment_elements:
                            prompt_order += element + " " + order2token_explicit[element] + " "
                    else:
                        prompt_order = permutation + " "
                    if self.ctrl_token == 'front':
                        input_seq_front = prompt_order + ' '.join(words[0])
                    else:
                        input_seq_front = ' '.join(words[0]) + ' ' + prompt_order

                prompt_order = ""
                aspect_label_latter = get_aspect_labels([absa_label[1]])[0]
                opinion_label_latter = get_opinion_labels([absa_label[1]])[0]
                if aspect_label_latter != 0 or opinion_label_latter != 0:
                    if self.virtual_token:
                        for element in sentiment_elements:
                            if element == "[A]" and aspect_label_latter != 0 and self.implicit_token:
                                prompt_order += "[A] <null> "
                            if element == "[O]" and opinion_label_latter != 0 and self.implicit_token:
                                prompt_order += "[O] <null> "
                            else:
                                prompt_order += element + " " + order2token_implicit[element] + " "
                    else:
                        prompt_order = permutation + " "
                    if self.ctrl_token == 'front':
                        input_seq_latter = prompt_order + ' '.join(words[1])
                    else:
                        input_seq_latter = ' '.join(words[1]) + ' ' + prompt_order
                else:
                    if self.virtual_token:
                        for element in sentiment_elements:
                            prompt_order += element + " " + order2token_explicit[element] + " "
                    else:
                        prompt_order = permutation + " "
                    if self.ctrl_token == 'front':
                        input_seq_latter = prompt_order + ' '.join(words[1])
                    else:
                        input_seq_latter = ' '.join(words[1]) + ' ' + prompt_order

                input_seq = input_seq_front + " </s> " + input_seq_latter
                # input_seq = ' '.join(words[0]) + "</s>" + ' '.join(words[1])

            absa_labels_seq = ABSA_format(absa_labels_seq)

            absa_prompt_len = self.args.absa_pre_seq_len

            tokenized_input_t5 = self.tokenizer_t5.batch_encode_plus(
                [input_seq],
                max_length=self.args.max_length * 2,
                # max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            label_max_length = 1024 if self.data_type == "test" else 512

            tokenized_absa_label = self.tokenizer_t5.batch_encode_plus(
                [absa_labels_seq],
                max_length=label_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            tokenized_absa_prompt = self.tokenizer_t5.batch_encode_plus(
                [absa_prompt],
                max_length=absa_prompt_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs_t5.append(tokenized_input_t5)

            self.absa_labels.append(tokenized_absa_label)

    def _get_tokenized(self, words, absa_label, is_label, is_prompt_tokenized):
        self.sents.append(' '.join(words))
        absa_labels_seq = absa_label
        self.is_labels.append(is_label)
        if is_label == 1:
            input_seq = "[A] <implicit_vtoken_a> [O] <implicit_vtoken_c> [S] <implicit_vtoken_s> [C] <implicit_vtoken_o> " + ' '.join(
                words)
        else:
            input_seq = "[O] <explicit_vtoken1> [C] <explicit_vtoken2> [S] <explicit_vtoken3> [A] <explicit_vtoken4> " + ' '.join(
                words)
        absa_labels_seq = ABSA_format(absa_labels_seq)
        absa_prompt = self.absa_prompt[12]
        absa_prompt_len = self.args.absa_pre_seq_len
        tokenized_input_t5 = self.tokenizer_t5.batch_encode_plus(
            [input_seq],
            max_length=self.args.max_length * 2,
            # max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt")

        label_max_length = 1024 if self.data_type == "test" else 512

        tokenized_absa_label = self.tokenizer_t5.batch_encode_plus(
            [absa_labels_seq],
            max_length=label_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")

        self.inputs_t5.append(tokenized_input_t5)
        self.absa_labels.append(tokenized_absa_label)
        # self.is_labels.append(is_label_classification)

    def _get_tokenized_multi(self, words, absa_label, is_label, is_prompt_tokenized):
        permutations = optim_orders_all[self.task_name][self.data_name][:self.topk]
        order2token_implicit = {"[A]": "<implicit_vtoken_a>", "[C]": "<implicit_vtoken_c>",
                                "[S]": "<implicit_vtoken_s>", "[O]": "<implicit_vtoken_o>"}
        order2token_explicit = {"[A]": "<explicit_vtoken1>", "[C]": "<explicit_vtoken2>", "[S]": "<explicit_vtoken3>",
                                "[O]": "<explicit_vtoken4>"}
        for permutation in permutations:
            self.is_labels.append((2, 2))
            sentiment_elements = permutation.split(" ")
            if self.virtual_token:
                prompt_order = ""
                for element in sentiment_elements:
                    prompt_order += element + " " + order2token_implicit[element] + " "
            else:
                prompt_order = permutation + " "
            input_seq = prompt_order + ' '.join(words)
            absa_labels_seq = ABSA_format_multi(absa_label, permutation)
            tokenized_input_t5 = self.tokenizer_t5.batch_encode_plus(
                [input_seq],
                max_length=self.args.max_length,
                # max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            label_max_length = 1024 if self.data_type == "test" else 512

            tokenized_absa_label = self.tokenizer_t5.batch_encode_plus(
                [absa_labels_seq],
                max_length=label_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs_t5.append(tokenized_input_t5)
            self.absa_labels.append(tokenized_absa_label)
            # self.is_labels.append(is_label_classification)
