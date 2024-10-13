from utils.parse_utils import parse_aste
import json


def parse_and_score(pred, target, data_args):
    all_labels, all_preds = [], []
    for i in range(len(pred)):
        pred_list = parse_aste(pred[i])
        all_preds.append(pred_list)
    for i in range(len(target)):
        gold_list = parse_aste(target[i])
        all_labels.append(gold_list)
    # if target_format == "AO":
    #     all_preds = distance_aware_merge_preds(all_preds)
    #     all_labels = all_labels[: int(len(all_labels) / 2)]
    all_preds = distance_aware_merge_preds(all_preds, data_args.topk)
    all_labels = distance_aware_merge_preds(all_labels, data_args.topk)
    raw_score = score(all_preds, all_labels, data_args)
    return raw_score


# def distance_aware_merge_preds(preds):
#     pred_nums = len(preds) // 2
#     merged_preds = []
#     for i in range(pred_nums):
#         ao_pred, oa_pred = preds[i], preds[i + pred_nums]
#         if ao_pred == oa_pred:
#             pred = ao_pred
#         else:
#             ao_pred = list([x for x in ao_pred])
#             oa_pred = list([x for x in oa_pred])
#             pred = []
#             ao_pred_dup, oa_pred_dup = ao_pred.copy(), oa_pred.copy()
#             # add the common triplet into ans (intersection set)
#             for cur in ao_pred:
#                 if cur in oa_pred_dup:
#                     ao_pred_dup.remove(cur)
#                     oa_pred_dup.remove(cur)
#                     pred.append(cur)
#         merged_preds.append(pred)
#     return merged_preds

def distance_aware_merge_preds(preds, top_K):
    group_size = top_K
    grouped_preds = [preds[i:i + group_size] for i in range(0, len(preds), group_size)]
    merged_preds = []
    for grouped_pred in grouped_preds:
        # # 初始化交集为第一个子列表的集合
        # intersection = set(grouped_pred[0])
        #
        # # 循环遍历剩余的子列表，计算交集
        # for sublist in grouped_pred[1:]:
        #     intersection &= set(sublist)

        # intersection = set(grouped_pred[0])
        # # 循环遍历剩余的子列表，计算并集
        # for sublist in grouped_pred[1:]:
        #     intersection = intersection.union(sublist)

        # 不按照交集 也不按照并集 count表示元组存在的个数
        tuple_counts = {}
        intersection = []
        for sublist in grouped_pred:
            # 使用集合存储唯一的元组
            unique_tuples = set(sublist)
            # 更新字典中每个元组的出现次数
            for tuple_item in unique_tuples:
                tuple_counts[tuple_item] = tuple_counts.get(tuple_item, 0) + 1
        for tuple_item, count in tuple_counts.items():
            # if count > top_K//2:
            if count > 14:
                intersection.append(tuple_item)
        merged_preds.append(intersection)
        # merged_preds.append(list(intersection))
    return merged_preds


# def score(all_preds, all_labels):
#     assert len(all_preds) == len(all_labels)
#     n_preds, n_labels, n_common = 0, 0, 0
#     for pred, label in zip(all_preds, all_labels):
#         n_preds += len(pred)
#         n_labels += len(label)
#         label_dup = label.copy()
#         for p in pred:
#             if p in label_dup:
#                 n_common += 1
#                 label_dup.remove(p)
#     if n_preds == 0:
#         precision = 0
#     else:
#         precision = n_common / n_preds
#     recall = n_common / n_labels
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     return {'precision': precision, "recall": recall, "f1_score": f1_score,
#             "n_preds": n_preds, "n_labels": n_labels, "n_common": n_common}


def score(all_preds, all_labels, data_args):
    assert len(all_preds) == len(all_labels)
    n_preds, n_labels, n_common = 0, 0, 0
    json_data = {"rest16": [], "result": {}}

    for pred, label in zip(all_preds, all_labels):
        is_exist = 0
        n_preds += len(pred)
        n_labels += len(label)
        label_dup = label.copy()
        for p in pred:
            if p in label_dup:
                n_common += 1
                is_exist = 1
                label_dup.remove(p)
        info_dict = {
            "labels": label,
            "preds": pred,
            "isExist": is_exist
        }
        json_data['rest16'].append(info_dict)

    precision = n_common / n_preds
    recall = n_common / n_labels
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    json_data['result'] = {'precision': precision, "recall": recall, "f1_score": f1_score,
                           "n_preds": n_preds, "n_labels": n_labels, "n_common": n_common}
    with open(
            f"./outputs_json/acos_rest16_seed3407_epoch20_bts16_lr1e-4_vt-True_it-True_ct-front_topk15_combined-True.json",
            'w',
            encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    return {'precision': precision, "recall": recall, "f1_score": f1_score,
            "n_preds": n_preds, "n_labels": n_labels, "n_common": n_common}
