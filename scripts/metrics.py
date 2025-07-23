import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string
import jailbreak_load
from collections import Counter
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm

def collect_csv_paths(base_dir): #instead of run it manualy
    csv_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                csv_paths.append(full_path)
    return csv_paths

metric_score = pd.DataFrame(columns=[])



def create_metric(metric_data, csv_file, keywords):
    df = pd.read_csv(f"{csv_file}")
    for index, row in df.iterrows():
        metric_data.loc[index, "jailbreak_prompt"] = row["jailbreak_prompt"]
        metric_data.loc[index, "source_layer"] = row["source_layer"]
        metric_data.loc[index, "target_layer"] = row["target_layer"]
        for keyword in keywords:
            output = row["output"] if pd.notna(row["output"]) else ""
            if re.search(re.escape(keyword), output, flags=re.IGNORECASE):
                metric_data.loc[index, f"{keyword}"] = 1
            else:
                metric_data.loc[index, f"{keyword}"] = 0
    return metric_data

def plot_keyword_heatmaps(metrics_df, output_dir=f"gemma-2-2b-it_gcg-evaluated-new-data_0"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    metadata_cols = {"jailbreak_prompt", "source_layer", "target_layer", "keyword_sum"}
    keyword_cols = [col for col in metrics_df.columns if col not in metadata_cols]
    metrics_df["keyword_sum"] = metrics_df[keyword_cols].sum(axis=1)

    # Count total matches per (source, target) pair
    heatmap_data = metrics_df.groupby(['source_layer', 'target_layer'])['keyword_sum'].sum().unstack(fill_value=0)
    #for keyword in keywords:

        #heatmap_data = metrics_df.groupby(['source_layer', 'target_layer'])[f"{keyword}"].sum().unstack(fill_value=0)


    # Normalize by number of keywords
    # Plot normalized heatmap
    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data.astype(int), annot=True, cmap="Blues", cbar=True)
    plt.title("Normalized Keyword Match Density per (Source, Target) Layer")
    plt.xlabel("Target Layer")
    plt.ylabel("Source Layer")
    filename = f"{output_dir}/heatmap.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_heat_maps(base_dir, metrics_df): #generate a heatmap of the metric for each jailbreak prompt
    files = collect_csv_paths(base_dir)
    metadata_cols = {"jailbreak_prompt", "source_layer", "target_layer", "keyword_sum"}
    keyword_cols = [col for col in metrics_df.columns if col not in metadata_cols]
    for i, file in enumerate(files):
        metrics = create_metric(metric_score, f"{file}", keyword_cols)
        plot_keyword_heatmaps(metrics, output_dir=f"/home/morg/students/yahavt/InSPEcT/full_dev_set/gemma-2-2b-it_gcg-evaluated-new-data_dev_{i}")

def create_keyword_dictionary(csv_file): #creates dict of counter of every word
    df = pd.read_csv(f"{csv_file}")
    keyword_counter = {}
    for i, out in enumerate(df['output']):
        if pd.notna(out) and isinstance(out, str):
            clean_out = out.translate(str.maketrans('', '', string.punctuation))
            output_list = clean_out.split()
            for word in output_list:
                if word in keyword_counter:
                    keyword_counter[word] +=1
                else:
                    keyword_counter[word] = 1
    return keyword_counter

def create_indicator_dictionary(csv_file):
    keyword_counter = create_keyword_dictionary(csv_file)
    indicator_dict = {key: 1 if value >= 1 else 0 for key, value in keyword_counter.items()}
    return indicator_dict


# Download stopwords once (if not already available)
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def plot_keyword_histogram(keyword_counter, output_path, top_n=None): #creates a histogram from the dict
    """
    Plots and saves a histogram from a word-count dictionary, removing English stopwords.
    
    Args:
        keyword_counter (dict): Dictionary of word:count pairs.
        output_path (str): File path to save the histogram image.
        top_n (int, optional): Number of top words to plot (by frequency). Defaults to all.
    """
    if not keyword_counter:
        print("Empty keyword counter. Nothing to plot.")
        return

    # Remove English stopwords
    stop_words = set(stopwords.words('english'))
    keyword_counter = {word: count for word, count in keyword_counter.items() if word.lower() not in stop_words}

    # Optionally select top N most common words
    if top_n:
        keyword_counter = dict(sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:top_n])

    words = list(keyword_counter.keys())
    counts = list(keyword_counter.values())

    if not words:
        print("No keywords left after removing stopwords.")
        return

    # Create the plot
    plt.figure(figsize=(max(10, len(words) * 0.4), 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Keyword Frequency Histogram (no stopwords)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    print(f"✅ Histogram saved to: {output_path}")


def plot_comparison_histogram(dict1, dict2, label1, label2, output_path, top_n=None):
    """
    Plots a grouped bar chart comparing two keyword frequency dictionaries.

    Args:
        dict1 (dict): First word:count dictionary.
        dict2 (dict): Second word:count dictionary.
        label1 (str): Label for the first dataset.
        label2 (str): Label for the second dataset.
        output_path (str): File path to save the histogram image.
        top_n (int): Number of top words to display (by absolute frequency difference).
    """
    # Remove English stopwords and lowercase all keys
    stop_words = set(stopwords.words('english'))
    dict1 = {k.lower(): v for k, v in dict1.items() if k.lower() not in stop_words}
    dict2 = {k.lower(): v for k, v in dict2.items() if k.lower() not in stop_words}

    # Union of all words from both dictionaries
    all_words = set(dict1.keys()).union(dict2.keys())

    # Calculate difference in counts
    diff = {word: dict1.get(word, 0) - dict2.get(word, 0) for word in all_words}

    # Sort by absolute difference to highlight biggest differences
    sorted_words = sorted(diff.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:top_n] if top_n else sorted_words
    words = [word for word, _ in top_words]

    # Get values from both dictionaries for those words
    values1 = [dict1.get(word, 0) for word in words]
    values2 = [dict2.get(word, 0) for word in words]

    # Plot
    bar_width = 0.35
    x = range(len(words))

    plt.figure(figsize=(max(10, len(words) * 0.6), 6))
    plt.bar([i - bar_width/2 for i in x], values1, width=bar_width, label=label1, color='skyblue')
    plt.bar([i + bar_width/2 for i in x], values2, width=bar_width, label=label2, color='sandybrown')

    plt.xticks(ticks=x, labels=words, rotation=45, ha='right')
    plt.ylabel('Frequency')
    plt.title('Keyword Frequency Comparison (no stopwords)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"✅ Comparison histogram saved to: {output_path}")
    return words


def list_of_keyword_dicts(): #emarge the dicts into one list of only weaks and one of only strongs
    data = jailbreak_load.load_jailbreak_data()
    weak_thershold = data.drop_duplicates("suffix_str")["univ_score"].quantile(.06)
    strong_threshold = data.drop_duplicates("suffix_str")["univ_score"].quantile(.94)
    list_of_weak_dicts = []
    list_of_strong_dicts = []
    csvs = collect_csv_paths("/home/morg/students/yahavt/InSPEcT/full_dev_set")

    for file in tqdm(csvs, total=len(csvs)):
        df = pd.read_csv(file)
        # dictionary = create_keyword_dictionary(file)
        dictionary = create_indicator_dictionary(file)
        univ_score = df[["jailbreak_prompt"]].rename(columns={"jailbreak_prompt": "suffix_str"}).merge(data)["univ_score"].max()
        if  univ_score >= strong_threshold:
            list_of_strong_dicts.append(dictionary)
        elif univ_score <= weak_thershold:
            list_of_weak_dicts.append(dictionary)

    merged_strong_counter = Counter()
    for dictionary in list_of_strong_dicts:
      merged_strong_counter.update(dictionary)

    merged_weak_counter = Counter()
    for dictionary in list_of_weak_dicts:
      merged_weak_counter.update(dictionary)


    return merged_strong_counter, merged_weak_counter

def top_500_words(): #takes the top 500 most common words in the dicts
    strong_counter, weak_counter = list_of_keyword_dicts()
    top_500_strong = strong_counter.most_common(500)
    top_500_weak = weak_counter.most_common(500)
    top_500_strong_dict = dict(top_500_strong)
    top_500_weak_dict = dict(top_500_weak)

    return strong_counter, weak_counter #top_500_strong_dict, top_500_weak_dict

strong_dict, weak_dict = top_500_words()
# plot_keyword_histogram(strong_dict, "/home/morg/students/yahavt/InSPEcT/full_dev_set/indicator_top_500_strong.png")
# plot_keyword_histogram(weak_dict, "/home/morg/students/yahavt/InSPEcT/full_dev_set/indicator_top_500_weak.png")
word_list = plot_comparison_histogram(strong_dict, weak_dict, "strong", "weak", "/home/morg/students/yahavt/InSPEcT/full_dev_set/indicator_top_500_compare.png", top_n=100)
def create_keywords(keywords):
    for word in keywords:
        metric_score[word] = []

create_keywords(word_list)
generate_heat_maps("/home/morg/students/yahavt/InSPEcT/full_dev_set", metric_score)