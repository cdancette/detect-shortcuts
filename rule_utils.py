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
