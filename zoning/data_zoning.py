import json
import os
import re
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch
#

def load_json_per_line(file_path):
   
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def split_text_preserve_punct(text):
 

    pattern = r'[^.?!]+[.?!]'  
    sentences_with_punct = re.findall(pattern, text)
 
    remainder = text
    for s in sentences_with_punct:
        
        remainder = remainder.split(s, 1)
        if len(remainder) >= 2:
            remainder = remainder[1]
        else:
            remainder = ""
    remainder = remainder.strip()
    if remainder:
        sentences_with_punct.append(remainder)

 
    seen = set()
    final_list = []
    for s in sentences_with_punct:
        ss = s.strip()
        if not ss:
            continue
        if ss not in seen:
            seen.add(ss)
            final_list.append(ss)
    return final_list

def build_full_article_sentences(data_per_line):


    tmp = {}
    for sec in data_per_line:
        art = sec['article_name']
        tmp.setdefault(art, []).append(sec)

   
    place_order = {
        "Introduction": 1,
        "Methods": 2,
        "Results": 3,
        "Conclusion": 4
    }

    articles_dict = {}
    for art_name, sections in tmp.items():
      
        sections_sorted = sorted(sections, key=lambda x: place_order.get(x['place'], 999))

      
        full_text = ""
        for sec in sections_sorted:

            text = sec['source_documents'][0].strip()
         
            full_text += text + " "

        full_text = full_text.strip()

        all_sentences = split_text_preserve_punct(full_text)
        sentence_to_section = [] 

        idx_map = []  
        for sec_idx, sec in enumerate(sections_sorted):
            sub_sent_list = split_text_preserve_punct(sec['source_documents'][0].strip())
            for _ in sub_sent_list:
                idx_map.append(sec_idx)

        for idx in idx_map:
            sentence_to_section.append(sections_sorted[idx]['place'])

        articles_dict[art_name] = {
            'ordered_sections': sections_sorted,
            'full_text': full_text,
            'full_sentences': all_sentences,
            'sentence_to_section': sentence_to_section
        }

    return articles_dict


def get_embeddings(sentences, model):

    embeddings = model.encode(sentences, convert_to_tensor=False, show_progress_bar=False)
    return np.array(embeddings)

def cluster_sentences(embeddings, n_clusters):
  
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans.labels_


def build_section_zone_data(art_name, article_info, n_clusters):

    sentences = article_info['full_sentences']          
    stat_to_sec = article_info['sentence_to_section']   


    actual_n_clusters = min(len(sentences), n_clusters)

   
    global sbert_model
    embeddings = get_embeddings(sentences, sbert_model)

 
    labels = cluster_sentences(embeddings, actual_n_clusters)  

 
    sec_zone_dict = {}
    for idx, sent in enumerate(sentences):
        section_place = stat_to_sec[idx]
        cluster_id = int(labels[idx]) + 1
        key = (section_place, cluster_id)
        sec_zone_dict.setdefault(key, []).append(sent)

   
    sec_meta = {}
    for sec in article_info['ordered_sections']:
        place = sec['place']
        sec_meta[(art_name, place)] = {
            "summary": sec["summary"],
            "video_file": sec["video_file"],
            "bounds": sec["bounds"],
            "video_duration": sec["video_duration"],
            "label":sec["label"]
        }

    new_sections_list = []      
    zone_txt_files = {}         

    
    for sec in article_info['ordered_sections']:
        place = sec['place']
        
        source_documents = []


        for cid in range(1, actual_n_clusters + 1):
            key = (place, cid)
            if key not in sec_zone_dict:
                continue
            sent_list = sec_zone_dict[key]
            if not sent_list:
                continue

            
            zone_text = " ".join(sent_list).strip()
            if not zone_text:
                continue
            source_documents.append(zone_text)

           
            article_folder = art_name
            if not os.path.exists(article_folder):
                os.makedirs(article_folder)
            zone_filename = f"{place}_zone{cid}.txt"
            zone_file_path = os.path.join(article_folder, zone_filename)
            with open(zone_file_path, 'w', encoding='utf-8') as f:
                f.write(zone_text)

            zone_txt_files.setdefault((art_name, place), []).append(zone_file_path)

        
        if source_documents:
            meta = sec_meta[(art_name, place)]
            new_sec_dict = {
                "source_documents": source_documents,
                "summary": meta["summary"],
                "article_name": art_name,
                "video_file": meta["video_file"],
                "bounds": meta["bounds"],
                "video_duration": meta["video_duration"],
                "place": place,
                "label": meta["label"]

            }
            new_sections_list.append(new_sec_dict)

    return new_sections_list, zone_txt_files


def main(input_json_path, output_json_path, n_clusters):
    
    raw_sections = load_json_per_line(input_json_path)

    
    articles_dict = build_full_article_sentences(raw_sections)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_path = 'mpnet'  
    global sbert_model
    sbert_model = SentenceTransformer(local_path, device=device)

    
    all_new_sections = []
    all_zone_files = {}
    for art_name, info in articles_dict.items():
        new_secs, zone_files = build_section_zone_data(art_name, info, n_clusters)
        all_new_sections.extend(new_secs)
        
        for k, v in zone_files.items():
            all_zone_files.setdefault(k, []).extend(v)

    
    with open(output_json_path, 'w', encoding='utf-8') as fout:
        for entry in all_new_sections:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write('\n')


   
    for (art, place), files in all_zone_files.items():
        for fp in files:
            print("   -", fp)


if __name__ == "__main__":
    
    input_json_path = "full-data.json"           
    output_json_path = "full-data-zoning.json"    
    n_clusters = 5                                     
    main(input_json_path, output_json_path, n_clusters)
