from typing import List
import re


# def parse_aste_a(seq):
#     triplets = []
#     sents = [s.strip() for s in seq.split('[SSEP]')]
#     for s in sents:
#         try:
#             _, a, b, c = s.split(":")
#             a, b, c = a.strip(), b.strip(), c.strip()
#             a = a.replace(', opinion', '')
#             b = b.replace(', sentiment', '')
#         except ValueError:
#             a, b, c = '', '', ''
#         triplets.append((a, b, c))
#     return triplets


def parse_aste_o(seq):
    triplets = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            _, a, b, c = s.split(":")
            a, b, c = a.strip(), b.strip(), c.strip()
            a = a.replace(', aspect', '')
            b = b.replace(', sentiment', '')
        except ValueError:
            a, b, c = '', '', ''
        triplets.append((b, a, c))
    return triplets


# def parse_aste(seq):
#     if seq.startswith('aspect'):
#         return parse_aste_a(seq)
#     else:
#         return parse_aste_o(seq)

def parse_aste(seq):
    triplets = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            a, b, c, d = '', '', '', ''
            a_match = re.search(r'\[A\] (.+?)(?=\[|$)', s)
            b_match = re.search(r'\[O\] (.+?)(?=\[|$)', s)
            c_match = re.search(r'\[S\] (.+?)(?=\[|$)', s)
            if '[C]' in s:
                d_match = re.search(r'\[C\] (.+?)(?=\[|$)', s)
                if a_match and b_match and c_match and d_match:
                    a = a_match.group(1).strip().replace(',', '')
                    b = b_match.group(1).strip().replace(',', '')
                    c = c_match.group(1).strip().replace(',', '')
                    d = d_match.group(1).strip().replace(',', '')
            else:
                if a_match and b_match and c_match:
                    a = a_match.group(1).strip().replace(',', '')
                    b = b_match.group(1).strip().replace(',', '')
                    c = c_match.group(1).strip().replace(',', '')
        except ValueError:
            a, b, c = '', '', ''
        if a == '' and b == '' and c == '':
            continue
        if '[C]' in s:
            triplets.append((a, d, c, b))
        else:
            triplets.append((a, b, c))
    return list(set(triplets))


def match_token_str(source: List, token: str):
    all_match_index, match_index = [], []
    begin_match = False
    for i in range(len(source)):
        cur = source[i]
        if token.startswith(cur) and not begin_match:
            match_token = token
            begin_match = True
            match_index.append(i)
            match_token = match_token.replace(cur, '', 1).strip()
        elif begin_match:
            if match_token[:len(cur)] == cur:
                match_index.append(i)
                match_token = match_token.replace(cur, '', 1).strip()
            else:
                begin_match = False
                match_index = []
                # 在这里匹配失败，但是不能放弃这个token
                if token.startswith(cur) and not begin_match:
                    match_token = token
                    begin_match = True
                    match_index.append(i)
                    match_token = match_token.replace(cur, '', 1).strip()
        if begin_match and match_token == '':
            begin_match = False
            all_match_index.append(match_index)
            match_index = []
    all_match_index = [tuple(x) for x in all_match_index]
    return all_match_index


def match_triplet_str(source, triplet):
    # closest !!!
    a, o = triplet[0], triplet[1]
    a_indexs = match_token_str(source, a)
    o_indexs = match_token_str(source, o)
    ans = []
    for a_index in a_indexs:
        for o_index in o_indexs:
            ans.append((
                abs(a_index[0] - o_index[0]),
                a_index, o_index, triplet[-1]
            ))
    ans.sort(key=lambda x: x[0])
    return ans[0]
