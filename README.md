# Rule-Mining for shortcut discovery

This repo contains the rule mining pipeline described in the article : 
Beyond Question-Based Biases: Assessing Multimodal Shortcut Learning in Visual Question Answering.

## Usage

### Installing gminer
First, you need to install gminer. Follow instructions at https://github.com/cdancette/GMiner.

### Visual Question Answering (VQA)

#### Download VQA and COCO data

First, run `./download.sh`. Data will be downloaded in the `./data` directory. 

#### Run the rule mining pipeline

Then run `python vqa.py --gminer_path <path_to_gminer>` to run our pipeline on the VQA v2 dataset.
You can change the parameters, see the end of the `vqa.py` file. 

This will save in logs/vqa2 various files containing the rules found in the dataset, 
the question_ids for easy and counterexamples splits, and the predictions made by the rule model.

To evaluate predictions, you can use the [multimodal](https://github.com/cdancette/multimodal) library: 

```bash
pip install multimodal
python -m multimodal vqa2-eval -p logs/vqa2/rules_predictions.json --split val
```


### Other task


#### fit
You can use our library to extract rule for any other dataset.

To do so, you can use the `fit` function in our `rule_mining.py``
It takes the following arguments : 
`fit(dataset, answer_ids, gminer_support=0.01, gminer_max_length=0, gminer_path=None)`, where : 

- `dataset` is a list of transactions. Each transaction is a list of integers describing tokens. 
- `answer_ids` is a list of integers, describing answer ids. They should be contained between 0 and max answer id.
- `gminer_support` is the minimum support used to mine frequent itemset.
- `gminer_max_length`: minimum length of an itemset. By default no minimum length
- `gminer_path`: path to the gminer binary you compiled (see top of the readme).


The function returns a list of rules, contained in namedtuples: `Rule = namedtuple("Rule", ["itemset", "ans", "sup", "conf"])`.

The itemset contains the input token ids, ans is the answer id, sup and conf are the support and the confidence of this rule.

#### match_rules

We provide a function to get, for each example in your dataset, all rules matching its input.

`match_rules(dataset, rules, answers=None, bsize=500)`

This will return `(matching_rules, correct_rules)`, where `matching_rules` is a list of the same length as the dataset, giving for each example, the matching rules. 

You can use this to build your counterexamples subset (examples where all rules are incorrect), or your easy subset (where at least one rule is correct).
