import numpy as np
import os
import random
import time
import json
import jsonlines
from transformers import LEDTokenizer
from gensim.models import KeyedVectors
from tqdm import tqdm
import multiprocessing
import functools
import spacy
import sys
import torch
import torch.nn.functional as F

sys.path.append("../../")
from utils.metrics import rouge
from model.dataloading import concatenate_documents, tokenize_tgs

def build_glove_embedding(tokenizer, glove_model_path, device="cuda"):
    print(f"Loading GloVe model from {glove_model_path}...")
    glove_kv = KeyedVectors.load(glove_model_path, mmap="r")
    emb_dim = glove_kv.vector_size
    vocab = list(tokenizer.get_vocab().keys())
    token2id = {tok: idx for idx, tok in enumerate(vocab)}
    weight = torch.zeros(len(vocab), emb_dim, dtype=torch.float32)
    for tok, idx in token2id.items():
        stripped = tok.replace("Ġ", "")
        if stripped and glove_kv.has_index_for(stripped):
            weight[idx] = torch.from_numpy(glove_kv[stripped])
    embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True).to(device)
    print("GloVe embeddings loaded into nn.Embedding on device", device)
    return embedding, token2id

def prepare_graph(concatenated_text, embedding, token2id, device="cuda", online=True, for_summary=False):
    docsep_token = "<doc-sep>"
    sentsep_token = "<sent-sep>"
    bos_token = "<s>"
    eos_token = "</s>"

    tokens_positions = []
    sents_positions = []
    docs_positions = []
    tokens = []
    token_idxs = []
    sents = []
    docs = []
    sent_tmp = []
    doc_tmp = []

    sent_token_contain_l = []
    sent_token_contain_r = []
    sent_token_contain_w = []
    doc_sent_contain_l = []
    doc_sent_contain_r = []
    doc_sent_contain_w = []

    for index, token in enumerate(concatenated_text):
        clean = token.replace("Ġ", "")
        if clean not in {docsep_token, sentsep_token, bos_token, eos_token}:
            tokens_positions.append(index)
            tokens.append(clean)
            token_idxs.append(token2id.get(token, 0))
            sent_tmp.append(clean)
        if token == sentsep_token:
            sents_positions.append(index)
            sents.append(sent_tmp)
            sent_index = len(sents)-1
            for i in range(len(token_idxs)-len(sent_tmp), len(token_idxs)):
                sent_token_contain_l.append(sent_index)
                sent_token_contain_r.append(i)
                sent_token_contain_w.append(1)
            doc_tmp.append(sent_tmp)
            sent_tmp = []
        if token == docsep_token:
            docs_positions.append(index)
            docs.append(doc_tmp)
            doc_index = len(docs)-1
            for i in range(len(sents)-len(doc_tmp), len(sents)):
                doc_sent_contain_l.append(doc_index)
                doc_sent_contain_r.append(i)
                doc_sent_contain_w.append(1)
            doc_tmp = []

    idx_tensor = torch.tensor(token_idxs, dtype=torch.long, device=device)
    emb = embedding(idx_tensor)
    emb_norm = F.normalize(emb, p=2, dim=1)
    sim_matrix = emb_norm @ emb_norm.t()
    N = sim_matrix.size(0)
    
    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=emb_norm.device)
    sims = sim_matrix[i_idx, j_idx]
    mask = sims > 0.5

    token_token_similarity_l = i_idx[mask].cpu().tolist()
    token_token_similarity_r = j_idx[mask].cpu().tolist()
    token_token_similarity_w = sims[mask].cpu().tolist()

    token_token_follow_l = list(range(N-1))
    token_token_follow_r = list(range(1, N))
    token_token_follow_w = [1]*(N-1)

    sents_stripped = [" ".join(s) for s in sents]
    sent_sent_rouge_l = []
    sent_sent_rouge_r = []
    sent_sent_rouge_w = []
    for i, si in enumerate(sents_stripped):
        for j, sj in enumerate(sents_stripped[i+1:], start=i+1):
            f1 = rouge(si, sj, types=["rouge2"])["rouge2"]["fmeasure"]
            sent_sent_rouge_l.append(i)
            sent_sent_rouge_r.append(j)
            sent_sent_rouge_w.append(f1)

    docs_stripped = ["\n".join([" ".join(s) for s in doc]) for doc in docs]
    doc_doc_rouge_l = []
    doc_doc_rouge_r = []
    doc_doc_rouge_w = []
    for i, di in enumerate(docs_stripped):
        for j, dj in enumerate(docs_stripped[i+1:], start=i+1):
            f1 = rouge(dj, di, types=["rougeLsum"])["rougeLsum"]["fmeasure"]
            doc_doc_rouge_l.append(i)
            doc_doc_rouge_r.append(j)
            doc_doc_rouge_w.append(f1)

    heterograph_data = {
        "token_token_similarity_l": token_token_similarity_l,
        "token_token_similarity_r": token_token_similarity_r,
        "token_token_similarity_w": token_token_similarity_w,
        "token_token_follow_l": token_token_follow_l,
        "token_token_follow_r": token_token_follow_r,
        "token_token_follow_w": token_token_follow_w,
        "sent_sent_rouge_l": sent_sent_rouge_l,
        "sent_sent_rouge_r": sent_sent_rouge_r,
        "sent_sent_rouge_w": sent_sent_rouge_w,
        "sent_token_contain_l": sent_token_contain_l,
        "sent_token_contain_r": sent_token_contain_r,
        "sent_token_contain_w": sent_token_contain_w,
        "tokens_positions": tokens_positions,
        "sents_positions": sents_positions,
    }
    if not for_summary:
        heterograph_data.update({
            "doc_sent_contain_l": doc_sent_contain_l,
            "doc_sent_contain_r": doc_sent_contain_r,
            "doc_sent_contain_w": doc_sent_contain_w,
            "doc_doc_rouge_l": doc_doc_rouge_l,
            "doc_doc_rouge_r": doc_doc_rouge_r,
            "doc_doc_rouge_w": doc_doc_rouge_w,
            "docs_positions": docs_positions,
        })
    if not online:
        heterograph_data["sents_tripped"] = sents_stripped
    return heterograph_data

def prepare_graph_multi_process_offline(i, samples, tokenizer, embedding, token2id, device):
    sample = samples[i]
    all_docs = sample["source_documents"]
    concatenated_text = concatenate_documents(
        all_docs, with_sent_sep=True, tokenizer=tokenizer, max_input_len=1024)
    heterograph_source = prepare_graph(
        concatenated_text, embedding, token2id, device=device, online=False, for_summary=False)

    tgt = sample["summary"]
    tokenized_tgt = tokenize_tgs(
        tgt, with_sent_sep=True, tokenizer=tokenizer, max_output_len=1024)
    heterograph_tgt = prepare_graph(
        tokenized_tgt, embedding, token2id, device=device, online=False, for_summary=True)

    sample["heterograph_source"] = heterograph_source
    sample["heterograph_tgt"] = heterograph_tgt
    return sample

if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')
    dataset_name = "full-data"
    pretrained_primer = 'primera'
    tokenizer = LEDTokenizer.from_pretrained(pretrained_primer, local_files_only=True)
    tokenizer.add_tokens(["<sent-sep>"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    glove_model_path = 'glove/glove-wiki-gigaword-100.model'
    embedding, token2id = build_glove_embedding(tokenizer, glove_model_path, device)

    save_file = f"data/{dataset_name}_graph_noun.json"
    if os.path.exists(save_file):
        os.remove(save_file)


    samples = []
    raw_data = []
    with jsonlines.open(f"datasets/{dataset_name}.json", "r") as reader:
        for obj in reader:
            raw_data.append(obj)
    for sample in raw_data:
        samples.append({
            "source_documents": sample.get("source_documents"),
            "summary": sample.get("summary"),
            "label": sample.get("label"),
            "article_name": sample.get("article_name"),
            "video_file": sample.get("video_file"),
            "bounds": sample.get("bounds"),
            "video_duration": sample.get("video_duration"),
            "place": sample.get("place"),
            "image_feat": sample.get("image_feat"),
            "audio_feat": sample.get("audio_feat")
        })

    print("dataset loaded", len(samples))

    random.seed(42)
    processes = 1
    chunksize = 100
    total = len(samples)
    for start in range(0, total, processes * chunksize):
        batch = samples[start:start + processes * chunksize]
        partial = functools.partial(
            prepare_graph_multi_process_offline,
            samples=batch,
            tokenizer=tokenizer,
            embedding=embedding,
            token2id=token2id,
            device=device
        )
        with multiprocessing.Pool(processes=processes) as p:
            results = list(tqdm(
                p.imap(partial, range(len(batch)), chunksize=chunksize),
                total=len(batch), desc="preparing graph data"
            ))
        with jsonlines.open(save_file, "a") as writer:
            writer.write_all(results)
    print("Done.")
