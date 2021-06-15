import os
from collections import namedtuple
import sys
import json
from typing import List
from tempfile import TemporaryDirectory

from tqdm import tqdm
import numpy as np
import torch

from rule_utils import superset_filtering

def loadjson(path):
    with open(path) as f:
        return json.load(f)


Rule = namedtuple("Rule", ["itemset", "ans", "sup", "conf"])

def run_gminer(transactions, support, max_length=0, gminer_path=None):
    with TemporaryDirectory() as tempdir:
        path_gminer_in = tempdir + f"/gminer_in.txt"
        path_gminer_out = tempdir + f"/gminer_out"

        # convert trans_by_ans to GMiner format
        print("Converting transactions to GMiner input format")
        if not os.path.exists(path_gminer_in):
            with open(path_gminer_in, "w") as f:
                for trans in tqdm(transactions):
                    trans = " ".join([str(x) for x in trans])
                    f.write(trans + "\n")

        print("Running GMiner")
        print(f"Number of transactions : {len(transactions)}")
        print(f"Number of items : {max([max(t) for t in transactions])}")

        if support * len(transactions) < 1:
            min_support = 1 / len(transactions)
            print(
                f"Warning: Number of transactions * support = {support * len(transactions)} is below 1. "
                f"Minimum support is {min_support}",
            )
            sys.exit(1)
        if gminer_path is None:
            gminer_path = "./GMiner"
        command = (
            f"{gminer_path} -i {path_gminer_in} -o {path_gminer_out} -s {support} -w 1"
        )
        if max_length != 0:
            command += f" -l {max_length}"

        print("Running Gminer:", command)
        out = os.system(command)
        if out != 0:
            os.remove(path_gminer_out)
            sys.exit(1)
        print("Done running gminer")

        itemsets = []
        print("Parsing Gminer output", path_gminer_out)
        with open(path_gminer_out, "r") as f:
            for line in tqdm(f):
                line = line.strip()
                tmp = line.split(" ")
                itemset = [int(x) for x in tmp[:-1]]
                supp = float(tmp[-1][:-1][1:])
                itemsets.append((itemset, supp))
        return itemsets


def fit(
    dataset,
    answer_ids,
    gminer_support=0.01,
    gminer_max_length=0,
    gminer_path=None,
):
    """
    train_dataset: list of token ids
    train_answers: list of answer ids
    """

    max_token_id = max(max(t) for t in dataset)
    answer_ids = [t + max_token_id + 1 for t in answer_ids]
    item_id_to_ans_id = {t: t - max_token_id - 1 for t in answer_ids}

    transactions = [
        items + [ans_id] for (items, ans_id) in zip(dataset, answer_ids)
    ]

    print(f"Minimum number of examples per rule: {gminer_support * len(transactions)}")

    itemsets = run_gminer(
        transactions,
        support=gminer_support,
        max_length=gminer_max_length,
        gminer_path=gminer_path,
    )

    supports_by_itemset = {}
    supports_by_itemset[()] = 1.0  # initialize empty tuple
    for itemset, support in itemsets:
        itemset = tuple(sorted(itemset))
        supports_by_itemset[itemset] = support

    # Extracting rules (itemsets with answers)
    pre_rules = []
    for (itemset, support_with_ans) in itemsets:
        for i, it in enumerate(itemset):
            if it in item_id_to_ans_id:
                del itemset[i]
                pre_rules.append(
                    (tuple(sorted(itemset)), item_id_to_ans_id[it], support_with_ans)
                )
                break
    print(f"Number of rules : {len(pre_rules)}")
    # Computing confidence on training set
    print("Computing confidences on training set")
    rules: List[Rule] = []
    for rule in tqdm(pre_rules):
        itemset, ans, support_with_ans = rule
        if len(itemset) == 0:
            confidence = support
        elif itemset in supports_by_itemset:
            # add confidence
            confidence = support_with_ans / supports_by_itemset[itemset]
        else:
            print(f"Missing data for itemset {itemset}...")
        rule = Rule(
            itemset=itemset, ans=ans, sup=supports_by_itemset[itemset], conf=confidence
        )
        rules.append(rule)

    ##########################
    # SUPERSET Filtering
    ##########################
    # Here, we remove an itemset if
    # there was a previous itemset that is
    # a subset of it, and had better conf
    # and the same answer
    # Also, if the itemset was previously in the rules,
    # then we discard it (it means that there was another
    # rule with another answer which has a better confidence).
    print("Performing superset filtering")
    rules = superset_filtering(rules)

    rules = sorted(
        rules, key=lambda r: (-r.conf, -r.sup, len(r.itemset))
    )  # conf, support, length

    print(f"Number of rules obtained from training set : {len(rules)}")
    return rules


def match_rules(
    dataset,
    rules: List[Rule],
    answers=None,
    bsize=500,
    stop_all_have_rules=False,
    stop_all_correct_rules=False,
):
    """
    This function will return lists of all rules that match a given example in the dataset.
    Args:
        dataset: list of list of token ids
        rules (List[Rule]): list of Rules
        answers: List[int]
    """
    # filling transaction matrix
    max_word_id = max(max(d) for d in dataset)
    transactions_matrix = np.zeros((len(dataset), max_word_id + 1), dtype=bool)
    for i, d in enumerate(dataset):
        transactions_matrix[i, d] = True

    transactions_matrix = torch.from_numpy(transactions_matrix).bool().cuda()
    pad_index = transactions_matrix.shape[1]
    N = transactions_matrix.shape[0]
    
    # pad index
    transactions_matrix = torch.cat(
        (transactions_matrix, torch.ones(N, 1).bool().cuda()), dim=1,
    )

    best_rules = dict()
    best_correct_rule = dict()
    all_rules = [[] for _ in range(len(transactions_matrix))]
    correct_rules = [[] for _ in range(len(transactions_matrix))]

    # Progress bars and iterables
    pbar = tqdm(total=len(transactions_matrix))
    pbar.set_description("Total rules found  ")
    pbar_correct = tqdm(total=len(transactions_matrix))
    pbar_correct.set_description("Correct rules found")
        
    for i in tqdm(range(0, len(rules), bsize), desc="Rules processed"):
        rs = rules[i : i + bsize]
        itemsets = [r.itemset for r in rs]
        max_length = max([len(r) for r in itemsets])
        itemsets = [list(r) + [pad_index] * (max_length - len(r)) for r in itemsets]
        indexes_concerned = (
            (transactions_matrix[:, itemsets].all(dim=2).nonzero())
            .detach()
            .cpu()
            .numpy()
        )  # (N * 2) where 2 = (trans_id, rule_id)
        transactions_for_rule = [[] for _ in range(len(rs))]

        num_trans_found = 0
        num_correct_trans_found = 0

        for j in range(len(indexes_concerned)):
            trans_id, rule_id = indexes_concerned[j]
            rule_id = rule_id + i
            rule = rules[rule_id]
            transactions_for_rule[rule_id - i].append(trans_id)
            if trans_id not in best_rules:
                num_trans_found += 1
                best_rules[trans_id] = rule
            all_rules[trans_id].append(rule)
            if rule.ans == answers[trans_id]:
                if trans_id not in best_correct_rule:
                    best_correct_rule[trans_id] = rule
                    num_correct_trans_found += 1
                correct_rules[trans_id].append(rule)

        pbar.update(num_trans_found)
        pbar_correct.update(num_correct_trans_found)

        if stop_all_have_rules and len(best_rules) == len(transactions_matrix):
            break
        if stop_all_correct_rules and len(best_correct_rule) == len(
            transactions_matrix
        ):
            break
    pbar.close()
    pbar_correct.close()
    del transactions_matrix

    return (
        all_rules,
        correct_rules,
    )
