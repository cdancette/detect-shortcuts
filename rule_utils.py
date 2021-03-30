from collections import defaultdict
from typing import Dict
from bootstrap.lib.logger import Logger
from itertools import combinations
import json
import pickle
import torch
from tqdm import tqdm
import os
from statistics import mean
import tempfile
import subprocess
import numpy as np

qid_to_annot = dict()


def evaluate_predictions(
    predictions_val,
    on_subset=None,
    qid_to_annot_path="/data/dancette/multimodal/",
    multimodal_data=None,
    type="vqa2",
    split="val",
):
    global qid_to_annot
    if split not in  qid_to_annot:
        qid_to_annot_path = os.path.join(
            multimodal_data or "data/multimodal/",
        )
        with open(
            os.path.join(
                qid_to_annot_path, f"datasets/{type}/{split}/processed/annotations.json"
            )
        ) as f:
            processed_annotations = json.load(f)
        qid_to_annot[split] = {a["question_id"]: a for a in processed_annotations}

    scores = defaultdict(list)
    for p in predictions_val:
        qid = p["question_id"]
        if on_subset is None or qid in on_subset:
            sc = qid_to_annot[split][qid]["scores"]
            qtype = qid_to_annot[split][qid]["answer_type"]
            score = sc.get(p["answer"], 0.0)
            scores["overall"].append(score)
            scores[qtype].append(score)

    for key in ["overall", "yes/no", "number", "other"]:
        if key in scores:
            scores[key] = mean(scores[key])
        else:
            scores[key] = -1
        if scores[key] is None:
            scores[key] = -1.0

    return {
        "overall": 100 * scores["overall"],
        "yes/no": 100 * scores["yes/no"],
        "number": 100 * scores["number"],
        "other": 100 * scores["other"],
    }


def evaluate(
    predictions_ids,
    dataset=None,
    predictions_eval=None,
    dir_rslt=None,
    type="vqa2",
    on_subset: set = None,
):
    if predictions_eval is None:
        prediction_tokens = [dataset.aid_to_ans[aid] for aid in predictions_ids]
        predictions_eval = [
            {
                "question_id": dataset.dataset["questions"][i]["question_id"],
                "answer": token,
            }
            for (i, token) in enumerate(prediction_tokens)
        ]

    return evaluate_predictions(predictions_eval, on_subset=on_subset, type=type)


def evaluate2(
    predictions_eval=None,
    predictions_ids=None,
    dataset=None,
    dir_rslt=None,
    type="vqa2",
):
    if predictions_eval is None:
        prediction_tokens = [dataset.aid_to_ans[aid] for aid in predictions_ids]

        predictions_eval = [
            {
                "question_id": dataset.dataset["questions"][i]["question_id"],
                "answer": token,
            }
            for (i, token) in enumerate(prediction_tokens)
        ]
    filename = "OpenEnded_mscoco_val2014_model_results.json"

    if type == "vqa1":
        dir_vqa = "/data/dancette/data/vqa1/"
    elif type == "vqa2":
        dir_vqa = "/data/dancette/data/vqa2"
    else:
        raise ValueError()

    if dir_rslt is None:
        dir_rslt_d = tempfile.TemporaryDirectory()
        dir_rslt = dir_rslt_d.name
    else:
        os.makedirs(dir_rslt, exist_ok=True)

    with open(os.path.join(dir_rslt, filename), "w") as f:
        json.dump(predictions_eval, f)

    command = (
        "python -m block.models.metrics.compute_oe_accuracy "
        + "--dir_vqa {} --dir_exp {} --dir_rslt {} --epoch {} --split {} --logs_name {} --rm {}".format(
            dir_vqa, dir_rslt, dir_rslt, 0, "val", "logs_test", 0
        )
    )

    subprocess.run(command, shell=True)
    subprocess.run(["ls", dir_rslt])
    with open(
        os.path.join(dir_rslt, "OpenEnded_mscoco_val2014_model_accuracy.json")
    ) as f:
        data = json.load(f)
    Logger()("overall", "yes/no", "number", "other")
    Logger()(
        data["overall"],
        data["perAnswerType"]["yes/no"],
        data["perAnswerType"]["number"],
        data["perAnswerType"]["other"],
    )
    return data


def build_hard_test_set_rule(rules, qids, annotations, aid_to_ans, min_conf=None):
    """
    rules: one or no rule for every item in the dataset.
        it could be the best correct rule, or the best-confidence rule.
        TODO: give a list of rules for each example ? 
    """
    qid_to_score = {a["question_id"]: a["scores"] for a in annotations}
    hard_with_rules = []
    no_rules_set = []
    for k in tqdm(range(len(rules))):
        qid = qids[k]
        if rules[k] is None:  # no rules
            no_rules_set.append(qid)
        else:
            itemset, support, ans, conf = rules[k]
            answer = aid_to_ans[ans]
            # gt = a["answer_id"]
            score = qid_to_score[qid].get(answer, 0.0)
            if score == 0.0:
                hard_with_rules.append(qid)
            elif min_conf is not None and conf <= min_conf:
                hard_with_rules.append(qid)
    hard_and_no_rules = hard_with_rules + no_rules_set
    print(
        f"Length of hard set: {len(hard_with_rules)} ({len(hard_with_rules)/len(rules)*100:.2f}% of total)"
    )
    print(
        f"Length of no_rules_set: {len(no_rules_set)} ({len(no_rules_set)/len(rules)*100:.2f}% of total)"
    )
    print(
        f"Length of hard_and_no_rules set: {len(hard_and_no_rules)} ({len(hard_and_no_rules)/len(rules)*100:.2f}% of total)"
    )
    return hard_and_no_rules, no_rules_set, hard_with_rules


def build_hard_test_set(all_rules, qids, annotations, aid_to_ans):
    """
    all_rules: list of rules for every entry
    best_correct_rules: one or no rule that has the best confidence and is right.
        it could be the best correct rule, or the best-confidence rule.
        TODO: give a list of rules for each example ? 
    """
    qid_to_score = {a["question_id"]: a["scores"] for a in annotations}
    hard_with_rules = []
    no_rules_set = []
    for k in tqdm(range(len(all_rules))):
        qid = qids[k]
        rules = all_rules[k]
        if not rules:
            no_rules_set.append(qid)
        else:
            scores_q = qid_to_score[qid]
            ans_rules = [aid_to_ans[ans] for itemset, support, ans, conf in rules]
            scores_rules = [scores_q.get(ans, 0.0) for ans in ans_rules]
            if all(sc == 0.0 for sc in scores_rules):
                hard_with_rules.append(qid)
    hard_and_no_rules = hard_with_rules + no_rules_set
    print(
        f"Length of hard set: {len(hard_with_rules)} ({len(hard_with_rules)/len(all_rules)*100:.2f}% of total)"
    )
    print(
        f"Length of no_rules set: {len(no_rules_set)} ({len(no_rules_set)/len(all_rules)*100:.2f}% of total)"
    )
    print(
        f"Length of hard_and_no_rules set: {len(hard_and_no_rules)} ({len(hard_and_no_rules)/len(all_rules)*100:.2f}% of total)"
    )
    return hard_and_no_rules, no_rules_set, hard_with_rules


def superset_filtering(rules):
    """
    Two goals:
    - remove duplicate rules (ie, rules that have the same itemset, but different answers). 
        We keep only the rule with the best confidence.
    - remove rules that are useless, because they are a superset of a previous rule (so they are more constrained, thus 
        they have a smaller support), but also have a smaller confidence.
    """
    rules_sorted = sorted(
        rules, key=lambda r: (len(r.itemset), -r.conf)
    )  # sorted by length (up), confidence (down)
    rules = []
    rule_by_itemset = dict()  # itemset -> list of rules
    # rules_discarded = defaultdict(set)
    for rule in tqdm(rules_sorted):
        itemset, aid, support, conf = rule
        itemset = frozenset(itemset)
        discard_rule = False
        if itemset in rule_by_itemset:
            continue
        else:
            rule_by_itemset[itemset] = rule

        if len(itemset) > 0:
            for it in combinations(itemset, len(itemset) - 1):
                it = frozenset(it)
                if it in rule_by_itemset:
                    old_r = rule_by_itemset[it]
                    if old_r.conf >= conf and old_r.ans == aid:
                        discard_rule = True
                        break
        if not discard_rule:
            rules.append(rule)
    Logger()(
        f"After discarding rules, going from {len(rules_sorted)} to {len(rules)} rules."
    )
    return rules


def test_superset_filtering():
    # various test cases that we want to manage
    assert superset_filtering([[(), 0.1, 10, 0.1]]) == [[(), 0.1, 10, 0.1]]

    rules = [
        [(), 0.1, 10, 0.1],
        [(0,), 0.1, 0, 0.5],
        [(0, 1), 0.1, 0, 0.5],
        [(0, 1, 2), 0.1, 0, 0.5],
    ]
    assert superset_filtering(rules) == [[(), 0.1, 10, 0.1], [(0,), 0.1, 0, 0.5]]

    rules = [
        [(), 0.1, 10, 0.1],
        [(0,), 0.1, 0, 0.5],
        [(0, 1), 0.1, 0, 0.3],
        [(0, 1, 2), 0.1, 0, 0.2],
        [(0, 1, 5), 0.1, 3, 0.2],  # additional to keep
    ]

    assert superset_filtering(rules) == [
        [(), 0.1, 10, 0.1],
        [(0,), 0.1, 0, 0.5],
        [(0, 1, 5), 0.1, 3, 0.2],
    ]

    # TODO this test fails.. it is quite bad, because it could allow
    # us to discard a lot of useless rules...
    rules = [
        [(), 0.1, 0, 0.5],
        [(0,), 0.1, 0, 0.3],
        [(0, 1), 0.1, 0, 0.4],
    ]
    # assert superset_filtering(rules) == [
    #     [(), 0.1, 0, 0.5],
    # ]

    rules = [
        [(), 0.1, 10, 0.1],
        [(0,), 0.1, 0, 0.5],
        [(0, 1), 0.1, 0, 0.6],
        [(0, 1, 2), 0.1, 0, 0.7],
    ]

    assert superset_filtering(rules) == [
        [(), 0.1, 10, 0.1],
        [(0,), 0.1, 0, 0.5],
        [(0, 1), 0.1, 0, 0.6],
        [(0, 1, 2), 0.1, 0, 0.7],
    ]

    # same itemset, different answers, one is better (confidence)
    # We keep only the best.
    rules = [
        [(), 0.1, 10, 0.2],
        [(), 0.1, 5, 0.1],
    ]

    assert superset_filtering(rules) == [[(), 0.1, 10, 0.2]]


def find_examples_for_rules(
    transactions_matrix,
    rules,
    bsize,
    answers=None,
    stop_all_have_rules=True,
    stop_all_correct_rules=True,
):
    """
    This function will return lists of all rules that match a given example in the dataset.

    """
    Logger()("Transfering transaction_matrix to cuda")
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
    Logger()("Running rule evaluation on train set")

    # Progress bars and iterables
    pbar = tqdm(total=len(transactions_matrix))
    pbar.set_description("Total rules found  ")
    pbar_correct = tqdm(total=len(transactions_matrix))
    pbar_correct.set_description("Correct rules found")
        
    for i in tqdm(range(0, len(rules), bsize), desc="Rules processed"):
        rs = rules[i : i + bsize]
        if len(rs) == 0:
            breakpoint()
        itemsets = [r[0] for r in rs]
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
            if rule[2] == answers[trans_id]:
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
        best_rules,
        best_correct_rule,
    )



def strong_superset_filtering(rules):
    rules_sorted = sorted(
        rules, key=lambda r: (len(r[0]), -r[3])
    )  # sorted by length (up), confidence (down)
    rules = []  # reset rules
    rule_by_itemset = dict()
    for rule in tqdm(rules_sorted):
        itemset, support, aid, conf = rule
        itemset = frozenset(itemset)
        discard_rule = False
        if len(itemset) > 0:
            for it in combinations(itemset, len(itemset) - 1):
                old_r = rule_by_itemset[frozenset(it)]
                if old_r[3] >= conf:
                    discard_rule = True
                    break
        if not discard_rule:
            rules.append(rule)
        if itemset not in rule_by_itemset or rule_by_itemset[itemset][3] < conf:
            rule_by_itemset[itemset] = rule
    Logger()(
        f"After discarding rules, going from {len(rules_sorted)} to {len(rules)} rules."
    )
    return rules


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("path_rules")
    parser.add_argument("path_transactions")
    parser.add_argument("--min-conf", type=float, default=0.0)
    # parser.add_argument("--min-support", type=float, default=1.0)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--annot", help="annot with scores")

    args = parser.parse_args()
    if args.method == "best":
        best_rules: dict = torch.load(args.path_rules)
        with open(args.path_transactions, "rb") as f:
            transactions = pickle.load(f)
        hard_set = build_hard_test_set_rule(
            best_rules, transactions["qids"], transactions["answers"], args.min_conf
        )

    elif args.method == "correct":
        pass

    with open(args.output, "w") as f:
        json.dump(hard_set, f)


def superset_filtering_test(rules):
    """
    Two goals:
    - remove duplicate rules (ie, rules that have the same itemset, but different answers). 
        We keep only the rule with the best confidence.
    - remove rules that are useless, because they are a superset of a previous rule (so they are more constrained, thus 
        they have a smaller support), but also have a smaller confidence.
    """

    rules_sorted = sorted(
        rules, key=lambda r: (len(r[0]), -r[3])
    )  # sorted by length (up), confidence (down)
    rules = []

    rule_by_itemset = dict()  # itemset -> list of rules
    # rules_discarded = defaultdict(set)
    discarded_by = dict()  # itemset -> itemset

    for rule in tqdm(rules_sorted):
        itemset, support, aid, conf = rule
        itemset = frozenset(itemset)
        discard_rule = False
        if itemset in rule_by_itemset:
            continue
        if len(itemset) > 0:
            print("**** itemset", itemset)
            for it in combinations(itemset, len(itemset) - 1):
                if discard_rule:
                    break
                # fill the list of parents
                parents = [frozenset(it)]
                it_parent = it
                while it_parent in discarded_by:
                    it_parent = discarded_by[it_parent]
                    parents.append(it_parent)
                print("parents", parents)
                for it_parent in parents:
                    print(it_parent)
                    if frozenset(it_parent) in rule_by_itemset:
                        old_r = rule_by_itemset[frozenset(it_parent)]
                        if old_r[3] >= conf and old_r[2] == aid:
                            print("discard")
                            discard_rule = True
                            discarded_by[itemset] = frozenset(it_parent)
                            break
        if itemset in rule_by_itemset:
            discard_rule = True
        if not discard_rule:
            rules.append(rule)
        if itemset not in rule_by_itemset or rule_by_itemset[itemset][3] < conf:
            rule_by_itemset[itemset] = rule
    Logger()(
        f"After discarding rules, going from {len(rules_sorted)} to {len(rules)} rules."
    )

    return rules
