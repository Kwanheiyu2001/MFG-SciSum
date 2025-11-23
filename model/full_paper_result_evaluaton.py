import os
import json
import glob
import re
from rouge_score import rouge_scorer
import pandas as pd


json_file_path = '11data-result/datamultinews_graph_noun_sentem.json'
generated_txt_folder = '11data-result/generated_txt_0_multinews_beam=1_1024_256'
output_csv_path = '11data-result/11-rouge_scores.csv'


def load_json_data(json_file_path):
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_original_summary(data, label='test'):
    original_summaries = {}
    for entry in data:
        if entry['label'] == label:
            article_name = entry['article_name']
            place = entry['place']
            summary = entry['summary']
            original_summaries.setdefault(article_name, {
                "Introduction": "",
                "Methods": "",
                "Results": "",
                "Conclusion": ""
            })
            if summary != "no summary provided":
                original_summaries[article_name][place] = summary
    return original_summaries

def get_generated_summary(generated_txt_folder):
    generated_summaries = {}

    pattern1 = re.compile(
        r"\('(?P<article>[^']+)',\)-\('(?P<section>Introduction|Methods|Results|Conclusion)',\)\.txt$"
    )
    pattern2 = re.compile(
        r"^(?P<article>.+?)-(?P<section>Introduction|Methods|Results|Conclusion)\.txt$"
    )

    for file_path in glob.glob(os.path.join(generated_txt_folder, '*.txt')):
        fname = os.path.basename(file_path)
        m = pattern1.match(fname) or pattern2.match(fname)
        if not m:
        
            continue

        art = m.group('article')
        sec = m.group('section')
        generated_summaries.setdefault(art, {
            "Introduction": "",
            "Methods": "",
            "Results": "",
            "Conclusion": ""
        })
        generated_summaries[art][sec] = open(file_path, 'r', encoding='utf-8').read().strip()

    return generated_summaries

def compute_rouge(orig, gen):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1','rouge2','rougeL','rougeLsum'], use_stemmer=True
    )
    s = scorer.score(orig, gen)
    return {
        'rouge-1-r': s['rouge1'].recall,
        'rouge-1-p': s['rouge1'].precision,
        'rouge-1-f': s['rouge1'].fmeasure,
        'rouge-2-r': s['rouge2'].recall,
        'rouge-2-p': s['rouge2'].precision,
        'rouge-2-f': s['rouge2'].fmeasure,
        'rouge-L-r': s['rougeL'].recall,
        'rouge-L-p': s['rougeL'].precision,
        'rouge-L-f': s['rougeL'].fmeasure,
        'rouge-Lsum-r': s['rougeLsum'].recall,
        'rouge-Lsum-p': s['rougeLsum'].precision,
        'rouge-Lsum-f': s['rougeLsum'].fmeasure
    }

def save_rouge_scores_to_csv(rouge_scores, output_csv_path):
    df = pd.DataFrame.from_dict(rouge_scores, orient='index')
    avg = df.mean().to_frame().T
    avg.index = ['avg_score']
    df = pd.concat([df, avg])
    df.to_csv(output_csv_path)

def main():

    data = load_json_data(json_file_path)
    original = get_original_summary(data, label='test')
    generated = get_generated_summary(generated_txt_folder)


    skipped = sorted(set(original) - set(generated))
    if skipped:
        print("跳过了以下文章（未找到任何生成摘要文件）：")
        for art in skipped:
            print(f"  - {art}")

    rouge_scores = {}
    for art, sections in original.items():
        orig_text = " ".join(sections[s] for s in ["Introduction","Methods","Results","Conclusion"] if sections[s])
        gen_sections = generated.get(art, {})
        gen_text = " ".join(gen_sections.get(s, "") for s in ["Introduction","Methods","Results","Conclusion"] if gen_sections.get(s))
        if orig_text and gen_text:
            rouge_scores[art] = compute_rouge(orig_text, gen_text)
   
            with open(f'11data-result/{art}-full-summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"Original Summary:\n{orig_text}\n\nGenerated Summary:\n{gen_text}\n")

    save_rouge_scores_to_csv(rouge_scores, output_csv_path)

if __name__ == "__main__":
    main()
