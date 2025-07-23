import argparse
import os
from src.patchscopes.layers import get_layers_combinations_for_model
from src.patchscopes.patch_layers import run_patching_and_save
from src.patchscopes.target_prompt import few_shot_demonstrations, create_jailbreak_prompts
from src.utils import get_model, get_tokenizer
import data_sets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model name', required=True)
    parser.add_argument('-d', '--dataset', type=str, help='the dataset name (dev or test)', required=True)
    parser.add_argument('-i', '--index', type=int, help='index of jailbreak prompt in dataset', required=True)
    # parser.add_argument('-t', '--target_prompt', type=int, help='target prompt prefix (e.g. "Paraphrase the following instructions:")', required=True)

    # parser.add_argument('-o', '--output_dir', type=str, help='output directory for results', required=False, nargs='?')

    args = parser.parse_args()
    model_path = args.model
    dataset_split = args.dataset
    index = args.index
    # target_prompt_prefix = args.target_prompt
    #output_dir = args.output_dir

    print(f"Model path: {model_path}", flush=True)
    tokenizer = get_tokenizer(model_path)
    model = get_model(model_path, to_device=True)
    layers_combinations = get_layers_combinations_for_model(model_path)

    # dataset = dev_set.create_dev_and_test_set()
    if dataset_split == "dev":
        dataset = data_sets.read_dev_set()
    elif dataset_split == "test":
        dataset = data_sets.read_test_set()

    model_name = model_path.split('/')[-1]
    dataset_name = "gcg-evaluated-new-data"

    # for i, (index, row) in enumerate(filtered_suffixes.iterrows()):
    row = dataset.iloc[index]
    jailbreak_prompt = row["suffix_str"]
    message = row["message_str"]

    output_dir = f"{model_name}_{dataset_name}_{dataset_split}_{index}" #f"{model_name}_{dataset_name}_{index}_{target_prompt_prefix.replace(' ', '_')}"
    print(f"Output directory: {output_dir}", flush=True)

    target_prompt = create_jailbreak_prompts(jailbreak_prompt, tokenizer) # add target prompt suffix before x's
    
    run_patching_and_save(
        model_path,
        jailbreak_prompt, # instead use the jailbreak prompt
        message,
        model, 
        tokenizer, 
        target_prompt,
        layers_combinations,
        output_dir,
    )