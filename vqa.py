from rule_mining import Rule, fit, match_rules
import numpy as np
import torch

import pickle
import os
from os import path as osp
from collections import Counter
import json
from typing import List
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer


def loadjson(path):
    with open(path) as f:
        return json.load(f)


def create_dataset(
    questions,
    visual_words="data/image_to_detection.json",
    annotations=None,
    textual=True,
    visual=True,
    visual_threshold=0.5,
    proportion=1.0,
    most_common_answers=None,
):
    print("Creating VQA binary dataset")
    visual_words = loadjson(visual_words)

    tokenizer = get_tokenizer("basic_english")

    total_len = int(len(questions) * proportion)
    transactions = []
    answers = []
    indexes = []

    skipped = 0

    #############
    # Regular textual
    #############
    for i in tqdm(range(total_len)):
        transaction = []

        if textual:
            # print("Adding textual words")
            # for i in range(total_len):
            tokens = tokenizer(questions[i]["question"])
            transaction.extend(tokens)

        if visual:
            # print("Adding visual words")
            # for i in range(total_len):
            image_id = str(questions[i]["image_id"])
            if image_id in visual_words:
                vwords = visual_words[image_id]
                classes = vwords["classes"]
                scores = vwords["scores"]
                if visual_threshold != 0:
                    classes = [
                        c
                        for (i, c) in enumerate(classes)
                        if scores[i] >= visual_threshold
                    ]
                classes = ["V_" + c for c in classes]  # visual marker
                transaction.extend(classes)
            else:
                skipped += 1
                continue

        transactions.append(transaction)
        indexes.append(i)
        if annotations is not None:
            answers.append(annotations[i]["multiple_choice_answer"])

    print("skipped:", skipped, "/", total_len)
    assert len(transactions) == len(answers)

    if annotations is not None and most_common_answers is not None:
        occurences = Counter(answers).most_common(most_common_answers)
        keep_answers = set(a for (a, _) in occurences)

        new_transactions = []
        new_answers = []
        new_indexes = []
        for k in range(len(transactions)):
            if answers[k] in keep_answers:
                new_transactions.append(transactions[k])
                new_answers.append(answers[k])
                new_indexes.append(indexes[k])
        transactions, answers, indexes = new_transactions, new_answers, new_indexes

    if annotations is not None:
        return transactions, answers, indexes

    return transactions, indexes


def vqa(
    textual=True,
    visual=True,
    visual_threshold=0.5,
    support_gminer=2e-5,
    gminer_path=None,
    min_conf=0.3,
    max_length=5,
    version="vqa2",
    save_dir=None,
    visual_words="data/image_to_detection.json",
):

    train_questions = loadjson(
        "data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json"
    )["questions"]
    train_annotations = loadjson("data/vqa2/v2_mscoco_train2014_annotations.json")[
        "annotations"
    ]

    os.makedirs(save_dir, exist_ok=True)
    train_dataset, train_answers, train_indexes = create_dataset(
        train_questions,
        annotations=train_annotations,
        proportion=1.0,
        most_common_answers=3000,
        textual=textual,
        visual=visual,
        visual_threshold=visual_threshold,
        visual_words=visual_words,
    )
    tokens = list(set(t for transaction in train_dataset for t in transaction))
    token_to_id = {t: i for (i, t) in enumerate(tokens)}
    train_transactions = [
        [token_to_id[t] for t in transaction] for transaction in train_dataset
    ]
    all_answers = list(set(train_answers))
    ans_to_id = {ans: i for (i, ans) in enumerate(all_answers)}
    train_answers_ids = [ans_to_id[ans] for ans in train_answers]

    # rule mining
    rules: List[Rule] = fit(
        train_transactions,
        train_answers_ids,
        gminer_support=support_gminer,
        gminer_max_length=max_length,
        gminer_path=gminer_path,
    )

    # - keep only rules with confidence > min_conf
    rules = [r for r in rules if r.conf >= min_conf]

    # show the best 20 rules
    for r in rules[:20]:
        print([tokens[tid] for tid in r.itemset], all_answers[r.ans], r.sup, r.conf)

    # match rules with examples
    matching_rules_train, matching_correct_rules_train = match_rules(
        train_transactions, rules, answers=train_answers_ids
    )

    # val
    val_questions = loadjson("data/vqa2/v2_OpenEnded_mscoco_val2014_questions.json")[
        "questions"
    ]
    val_annotations = loadjson("data/vqa2/v2_mscoco_val2014_annotations.json")[
        "annotations"
    ]
    val_dataset, val_answers, val_indexes = create_dataset(
        val_questions,
        annotations=val_annotations,
        proportion=1.0,
        textual=textual,
        visual=visual,
        visual_threshold=visual_threshold,
        visual_words=visual_words,
    )

    val_transactions = [
        [token_to_id[t] for t in transaction if t in token_to_id]
        for transaction in val_dataset
    ]
    val_answers_ids = [ans_to_id.get(ans, -1) for ans in val_answers]

    matching_rules_val, matching_correct_rules_val = match_rules(
        val_transactions, rules, val_answers_ids
    )

    # - create hard evaluations set
    # we load annotations, because we'll consider every answer, not only the top answer.
    qid_counterexamples = []
    qid_easy = []
    qid_hard = []
    for annot, rs in zip(val_annotations, matching_rules_val):
        possible_answers = set(ans["answer"] for ans in annot["answers"])
        rules_answers = set(all_answers[r.ans] for r in rs)
        if len(set.intersection(rules_answers, possible_answers)) == 0 and len(rs) != 0:
            # goes into counterexamples
            qid_counterexamples.append(annot["question_id"])
        elif len(rs) == 0:
            qid_hard.append(annot["question_id"])
        else:
            qid_easy.append(annot["question_id"])

    # keep_rules:
    # we keep only one correct rule per training example
    keep_rules = set()
    for rs in matching_correct_rules_train:
        if rs:
            keep_rules.add(rs[0])

    # build predictions on validation set.
    predictions = []
    for i, rs in enumerate(matching_rules_val):
        qid = val_questions[val_indexes[i]]["question_id"]
        rs = [r for r in rs if r in keep_rules]
        if rs:
            ans = all_answers[rs[0].ans]
        else:
            ans = "yes"
        predictions.append(
            {"question_id": qid, "answer": ans,}
        )

    # save predictions and rules
    with open(os.path.join(save_dir, "rules_predictions.json"), "w") as f:
        json.dump(predictions, f)
    with open(os.path.join(save_dir, "rules.pickle"), "bw") as f:
        pickle.dump(rules, f)
    with open(os.path.join(save_dir, "counterexamples.json"), "w") as f:
        json.dump(qid_counterexamples, f)
    with open(os.path.join(save_dir, "easy.json"), "w") as f:
        json.dump(qid_easy, f)
    with open(os.path.join(save_dir, "hard.json"), "w") as f:
        json.dump(qid_hard, f)

    return rules, qid_easy, qid_counterexamples, qid_hard


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--support", default=2.0e-5, type=float)
    parser.add_argument("--max_length", default=5, type=int)
    parser.add_argument("--min_conf", default=0.3, type=float)
    parser.add_argument("--gminer_path")
    parser.add_argument("--visual_words", default="data/image_to_detection.json")
    args = parser.parse_args()

    (rules, qid_easy, qid_counterexamples, qid_hard) = vqa(
        support_gminer=args.support,
        max_length=args.max_length,
        min_conf=args.min_conf,
        save_dir=args.save_dir,
        gminer_path=args.gminer_path,
        visual_words=args.visual_words,
    )
