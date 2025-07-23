from jailbreak_load import load_jailbreak_data
import pandas as pd


def read_dev_set():
    return pd.read_csv("data/dev_set.csv")

def read_test_set():
    return pd.read_csv("data/test_set.csv")

def create_dev_and_test_set():
    data = load_jailbreak_data("gemma2")
    d = data.drop_duplicates("suffix_str")
    strong_threshold = data.drop_duplicates("suffix_str")["univ_score"].quantile(.94)
    weak_threshold = data.drop_duplicates("suffix_str")["univ_score"].quantile(.06)

    full_weak_set = d[d["univ_score"] <= weak_threshold] 
    full_strong_set = d[d["univ_score"] >= strong_threshold]

    full_set = pd.concat([full_weak_set, full_strong_set])

    weak_dev = full_weak_set.sample(28, random_state=42)
    strong_dev = full_strong_set.sample(28, random_state=42)
    dev_set = pd.concat([weak_dev, strong_dev])
    dev_set.to_csv("data/dev_set.csv", index=False)

    test_set = full_set.merge(dev_set, on="suffix_str", how='left', indicator=True)
    test_set = test_set[test_set['_merge'] == 'left_only'].drop(columns=['_merge'])
    test_set = test_set[["suffix_str"]]
    test_set = full_set.merge(test_set, on="suffix_str")
    test_set.to_csv("data/test_set.csv", index=False)

read_test_set()