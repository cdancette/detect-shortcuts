from tempfile import TemporaryDirectory
from rule_mining import fit, Rule, match_rules
# from rule_utils import match_rules

def test_fit():
    dataset = [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 4],
    ]
    answers = [0, 0, 1, 1, 2, 2]
    rules = fit(dataset, answers, support_gminer=0.2)

    assert len(rules) == 4
    for k, (itemset, ans) in enumerate([[(2,), 0], [(3,), 1], [(4,), 2], [(), 0],]):
        assert rules[k].itemset == itemset
        assert rules[k].ans == ans

    # match_rules(dataset, rules)
    all_rules, correct_rules = match_rules(dataset, rules, answers=answers, bsize=10)
    print(all_rules)

    # item 0
    assert len(all_rules[0]) == 2
    assert all_rules[0][0].itemset == (2,)
    assert all_rules[0][1].itemset == ()
    assert len(correct_rules[0]) == 2
    assert correct_rules[0][0].itemset == (2,)
    assert correct_rules[0][1].itemset == ()

    # item 2
    assert len(all_rules[2]) == 2
    assert all_rules[2][0].itemset == (3,)
    assert all_rules[2][1].itemset == ()
    assert len(correct_rules[2]) == 1
    assert correct_rules[2][0].itemset == (3,)
