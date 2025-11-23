from torch.utils.data import DataLoader, Dataset
import torch
import random
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import sys
import os
from nltk import sent_tokenize

sys.path.append("../")
#sys.path.append("../../")
from graph.building_graph import build_graph
from datasets import load_dataset, load_from_disk, Features, Value, Sequence


def concatenate_documents(all_docs, with_sent_sep, tokenizer, max_input_len):
   
    for i, doc in enumerate(all_docs):
        doc = doc.replace("\n", " ").strip()
        doc = " ".join(doc.split()) 
        all_docs[i] = doc

    
    max_doc_len = max_input_len // len(all_docs)
    tokenized_text = []
    for doc in all_docs:
        
        doc_words = doc.split()
        if len(doc_words) > max_doc_len:
            doc = " ".join(doc_words[:max_doc_len])
        if with_sent_sep:
            sents = sent_tokenize(doc)
            doc = " ".join([sent + " <sent-sep>" for sent in sents])
        if len(all_docs) > 1:
            tokenized_text.extend(tokenizer.tokenize(doc)[:max_doc_len - 2])
        else:
            tokenized_text.extend(tokenizer.tokenize(doc)[:max_doc_len - 3])
        tokenized_text.append("<doc-sep>")
    tokenized_text = [tokenizer.bos_token] + tokenized_text + [tokenizer.eos_token]
    return tokenized_text


def tokenize_tgs(tgt, with_sent_sep, tokenizer, max_output_len):
    if with_sent_sep:
        sents = sent_tokenize(tgt)
        tgt = " ".join([sent + " <sent-sep>" for sent in sents])
    tokenized_text = tokenizer.tokenize(tgt)
    if max_output_len > 0:
        tokenized_text = tokenized_text[:max_output_len - 2]
    tokenized_text = [tokenizer.bos_token] + tokenized_text + [tokenizer.eos_token]
    return tokenized_text


class SummarizationDataset(Dataset):
    def __init__(
            self,
            dataset,
            dataset_name,
            with_sent_sep,
            tokenizer,
            max_input_len,
            max_output_len,
            mask_num=5,
            dataset_type="train",
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.with_sent_sep = with_sent_sep
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        self.sentsep_token_id = self.tokenizer.convert_tokens_to_ids("<sent-sep>")
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        self.dataset_type = dataset_type

        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        
        all_docs = entry["source_documents"]
        concatenated_source = concatenate_documents(all_docs, self.with_sent_sep, self.tokenizer, self.max_input_len)
        input_ids_source = self.tokenizer.convert_tokens_to_ids(concatenated_source)

       
        heterograph_data_source = entry["heterograph_source"]
        tokens_positions_source = torch.tensor(heterograph_data_source["tokens_positions"])
        sents_positions_source = torch.tensor(heterograph_data_source["sents_positions"])
        docs_positions_source = torch.tensor(heterograph_data_source["docs_positions"])
        heterograph_source = build_graph(heterograph_data=heterograph_data_source, for_summary=False)
        
        article_name = entry["article_name"]
        video_file = entry["video_file"]
        bounds = entry["bounds"]
        video_duration = entry["video_duration"]
        image_feat = entry["image_feat"]
        audio_feat = entry["audio_feat"]
        place = entry["place"]

        
        tgt = entry["summary"]

        tokenized_tgs = tokenize_tgs(tgt, False, self.tokenizer, self.max_output_len)
        output_ids = self.tokenizer.convert_tokens_to_ids(tokenized_tgs)

        input_ids_summary = self.tokenizer.convert_tokens_to_ids(
            tokenize_tgs(tgt, True, self.tokenizer, self.max_output_len))

        heterograph_data_tgt = entry["heterograph_tgt"]
        tokens_positions_tgt = torch.tensor(heterograph_data_tgt["tokens_positions"])
        sents_positions_tgt = torch.tensor(heterograph_data_tgt["sents_positions"])
        heterograph_tgt = build_graph(heterograph_data=heterograph_data_tgt, for_summary=True)
        

        if self.dataset_type == "train":
            return torch.tensor(input_ids_source), torch.tensor(
                output_ids), torch.tensor(
                input_ids_summary), heterograph_source, tokens_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, tokens_positions_tgt, sents_positions_tgt, article_name, video_file, bounds, video_duration, image_feat, audio_feat, place
        else:
            return torch.tensor(input_ids_source), torch.tensor(
                output_ids), torch.tensor(
                input_ids_summary), heterograph_source, tokens_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, tokens_positions_tgt, sents_positions_tgt, tgt, article_name, video_file, bounds, video_duration, image_feat, audio_feat, place
        print("finish")


def collate_fn(batch):
    
    last_token = batch[0][0][-1].item()
    if last_token == 2:
        pad_token_id = 1  
    elif last_token == 1:
        pad_token_id = 0  
    else:
        raise ValueError(f"Unexpected last token id: {last_token}")

    sample_len = len(batch[0])
    if sample_len == 17:
        
        (input_ids_source,
         output_ids,
         input_ids_summary,
         heterograph_source,
         words_positions_source,
         sents_positions_source,
         docs_positions_source,
         heterograph_tgt,
         words_positions_tgt,
         sents_positions_tgt,
         article_name,
         video_file,
         bounds,
         video_duration,
         image_feat,
         audio_feat,
         place) = zip(*batch)
        is_train = True
    elif sample_len == 18:
        
        (input_ids_source,
         output_ids,
         input_ids_summary,
         heterograph_source,
         words_positions_source,
         sents_positions_source,
         docs_positions_source,
         heterograph_tgt,
         words_positions_tgt,
         sents_positions_tgt,
         tgt,
         article_name,
         video_file,
         bounds,
         video_duration,
         image_feat,
         audio_feat,
         place) = zip(*batch)
        is_train = False
    else:
        raise ValueError(f"Unexpected batch element length: {sample_len}")

    
    input_ids_source = torch.nn.utils.rnn.pad_sequence(
        input_ids_source, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    input_ids_summary = torch.nn.utils.rnn.pad_sequence(
        input_ids_summary, batch_first=True, padding_value=pad_token_id
    )

    if is_train:
        return (
            input_ids_source,
            output_ids,
            input_ids_summary,
            heterograph_source,
            words_positions_source,
            sents_positions_source,
            docs_positions_source,
            heterograph_tgt,
            words_positions_tgt,
            sents_positions_tgt,
            article_name,
            video_file,
            bounds,
            video_duration,
            image_feat,
            audio_feat,
            place

        )
    else:
        return (
            input_ids_source,
            output_ids,
            input_ids_summary,
            heterograph_source,
            words_positions_source,
            sents_positions_source,
            docs_positions_source,
            heterograph_tgt,
            words_positions_tgt,
            sents_positions_tgt,
            tgt,  
            article_name,
            video_file,
            bounds,
            video_duration,
            image_feat,
            audio_feat,
            place
        )




features = Features({
    "source_documents": Sequence(Value("string")),
    "summary": Value("string"),
    "article_name": Value("string"),
    "video_file": Value("string"),
    "bounds": Sequence(Value("float32")),
    "video_duration": Value("float32"),
    "place": Value("string"),
    "label": Value("string"),
    "image_feat": Sequence(Value("float32")),
    "audio_feat": Sequence(Value("float32")),

    "heterograph_source": {
        "token_token_similarity_l": Sequence(Value("int64")),
        "token_token_similarity_r": Sequence(Value("int64")),
        "token_token_similarity_w": Sequence(Value("float64")),
        "token_token_follow_l": Sequence(Value("int64")),
        "token_token_follow_r": Sequence(Value("int64")),
        "token_token_follow_w": Sequence(Value("int64")),
        "sent_sent_rouge_l": Sequence(Value("int64")),
        "sent_sent_rouge_r": Sequence(Value("int64")),
        "sent_sent_rouge_w": Sequence(Value("float64")),
        "sent_token_contain_l": Sequence(Value("int64")),
        "sent_token_contain_r": Sequence(Value("int64")),
        "sent_token_contain_w": Sequence(Value("int64")),
        "tokens_positions": Sequence(Value("int64")),
        "sents_positions": Sequence(Value("int64")),
        "docs_positions": Sequence(Value("int64")),
        "sents_tripped": Sequence(Value("string")),
        "sent_sent_similarity_l": Sequence(Value("int64")),
        "sent_sent_similarity_r": Sequence(Value("int64")),
        "sent_sent_similarity_w": Sequence(Value("float64")),
        "sents_embeddings": Sequence(Sequence(Value("float64"))),
        
        "doc_sent_contain_l": Sequence(Value("int64")),
        "doc_sent_contain_r": Sequence(Value("int64")),
        "doc_sent_contain_w": Sequence(Value("int64")),
        "doc_doc_rouge_l": Sequence(Value("int64")),
        "doc_doc_rouge_r": Sequence(Value("int64")),
        "doc_doc_rouge_w": Sequence(Value("float64"))
    },
    "heterograph_tgt": {
        "token_token_similarity_l": Sequence(Value("int64")),
        "token_token_similarity_r": Sequence(Value("int64")),
        "token_token_similarity_w": Sequence(Value("float64")),
        "token_token_follow_l": Sequence(Value("int64")),
        "token_token_follow_r": Sequence(Value("int64")),
        "token_token_follow_w": Sequence(Value("int64")),
        "sent_sent_rouge_l": Sequence(Value("int64")),
        "sent_sent_rouge_r": Sequence(Value("int64")),
        "sent_sent_rouge_w": Sequence(Value("float64")),
        "sent_token_contain_l": Sequence(Value("int64")),
        "sent_token_contain_r": Sequence(Value("int64")),
        "sent_token_contain_w": Sequence(Value("int64")),
        "tokens_positions": Sequence(Value("int64")),
        "sents_positions": Sequence(Value("int64")),
        "sents_tripped": Sequence(Value("string")),
        "sent_sent_similarity_l": Sequence(Value("int64")),
        "sent_sent_similarity_r": Sequence(Value("int64")),
        "sent_sent_similarity_w": Sequence(Value("float64")),
        "sents_embeddings": Sequence(Sequence(Value("float64")))
    }
})


def get_dataloader_summ(args, tokenizer, split_name, num_workers, is_shuffle):
    
    cache_dir = os.path.join(args.data_path, f"{args.dataset_name}_{split_name}_hfds")

    if os.path.exists(cache_dir):
        
        print(f"Loading HF-Dataset for {split_name} from {cache_dir}")
        dataset = load_from_disk(cache_dir)
    else:
        
        print("Loading raw JSON dataset …")
        dataset_all = load_dataset(
            'json',
            data_files=args.data_path + f'{args.dataset_name}_graph_noun_sentem.json',
            split='all',
            features=features,
        )
        print("Total examples:", len(dataset_all))

        
        if split_name == "train":
            dataset = dataset_all.filter(lambda s: s['label'] == 'train')
            if 0 < args.num_train_data < len(dataset):
                import random
                random.seed(args.rand_seed)
                dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_train_data))

        elif split_name == "validation":
            dataset = dataset_all.filter(lambda s: s['label'] == 'val')

        elif split_name == "test":
            dataset = dataset_all.filter(lambda s: s['label'] == 'test')
            if 0 < args.num_test_data < len(dataset):
                import random
                random.seed(args.rand_seed)
                dataset = dataset.select(random.choices(range(len(dataset)), k=args.num_test_data))

        else:
            raise ValueError(f"Unknown split name: {split_name}")

        print(f"{split_name} size after filter/select:", len(dataset))
        
        print(f"Saving HF-Dataset for {split_name} to {cache_dir}")
        dataset.save_to_disk(cache_dir)

    
    summarization_dataset = SummarizationDataset(
        dataset=dataset,
        dataset_name=args.dataset_name,
        with_sent_sep=args.with_sent_sep,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        mask_num=args.mask_num,
        dataset_type=split_name,
    )

    return DataLoader(
        summarization_dataset,
        batch_size=args.batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

