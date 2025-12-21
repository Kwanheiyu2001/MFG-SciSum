import pandas as pd
import pdb
import json
import os
import argparse
import torch
#from torchmetrics.text.rouge import ROUGEScore
import os, pickle
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tokenization import LEDTokenizer
from mfgmodeling import LEDForConditionalGeneration
from dataloading_mfg import get_dataloader_summ

import sys

sys.path.append("../../")
#from utils.metrics import rouge
from rouge_score import rouge_scorer  # HuggingFace 的 rouge_score
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """
    This function is borrowed from fairseq.
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


class HGSummarizer(pl.LightningModule):
    def __init__(self, args):
        super(HGSummarizer, self).__init__()
        self.args = args

        model_name_or_path=args.pretrained_primer

        self.tokenizer = LEDTokenizer.from_pretrained(model_name_or_path)
        self.model = LEDForConditionalGeneration.from_pretrained(args.pretrained_primer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_ddp = self.args.speed_strategy == "ddp"
        self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        # or use eos_token as sent-sep
        self.tokenizer.add_tokens(["<sent-sep>"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.sentsep_token_id = self.tokenizer.convert_tokens_to_ids("<sent-sep>")
        self.scorer = rouge_scorer.RougeScorer(
                       ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
        if getattr(self.args, "grad_ckpt", False):
              self.model.gradient_checkpointing_enable()

    def forward(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                sents_positions_source,
                docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt,article_name,
                video_file,bounds,video_duration,image_feat,audio_feat,place):
        device = input_ids_source.device
        # print("encoding source", len(heterograph_source))
        decoder_input_ids = output_ids[:, :-1]
        # get the input ids and attention masks together
        global_attention_mask_source = torch.zeros_like(input_ids_source).to(device)
        # put global attention on <s> token
        global_attention_mask_source[:, 0] = 1
        # put global attention on <doc-sep> token
        global_attention_mask_source[input_ids_source == self.docsep_token_id] = 1
        # put global attention on <sent-sep> token
        global_attention_mask_source[input_ids_source == self.sentsep_token_id] = 1

        attention_mask = torch.ones_like(input_ids_source).to(device)
        attention_mask = attention_mask.type_as(input_ids_source)
        attention_mask[input_ids_source == self.pad_token_id] = 0
        # encoding source documents
        outputs_source = self.model(
            input_ids_source,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            global_attention_mask=global_attention_mask_source,
            use_cache=False,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source,
            article_name=article_name,
            video_file=video_file,
            bounds=bounds,
            video_duration=video_duration,
            image_feat=image_feat,
            audio_feat=audio_feat,
            place=place
        )
        lm_logits = outputs_source.logits
        assert lm_logits.shape[-1] == self.model.config.vocab_size

    
        attention_mask_summary = torch.ones_like(input_ids_summary).to(device)
        attention_mask_summary[input_ids_summary == self.pad_token_id] = 0

        global_attention_mask_summary = torch.zeros_like(input_ids_summary).to(device)
        global_attention_mask_summary[:, 0] = 1
        global_attention_mask_summary[input_ids_summary == self.sentsep_token_id] = 1

        
        outputs_summary = self.model(
            input_ids=input_ids_summary.to(device),
            attention_mask=attention_mask_summary,  
            decoder_input_ids=None,
            global_attention_mask=global_attention_mask_summary,
            use_cache=False,
            heterograph=heterograph_tgt,
            words_positions_source=words_positions_tgt,
            sents_positions_source=sents_positions_tgt,
            docs_positions_source=None,
            article_name=article_name,
            video_file=video_file,
            bounds=bounds,
            video_duration=video_duration,
            image_feat=image_feat,
            audio_feat=audio_feat,
            place=place
        )

        return lm_logits, outputs_source.mgat_outputs, outputs_source.sagpooling_outputs, outputs_summary.mgat_outputs
        
    def lr_scheduler_step(self, scheduler, optimizer, metric=None):
            scheduler.step()



    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]




    def shared_step(self, input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source,
                    sents_positions_source,
                    docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt,article_name,video_file,bounds,video_duration,image_feat,audio_feat,place):

        lm_logits, mgat_outputs_source, sagpooling_ouputs, mgat_outputs_summary = self.forward(input_ids_source,
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
                                                                                               audio_feat,place)

        
        cos = torch.nn.CosineSimilarity(dim=1)
        graph_sim = torch.mean(cos(sagpooling_ouputs, mgat_outputs_summary))
       
        labels = output_ids[:, 1:].clone()

        if self.args.label_smoothing == 0.0:
            
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        if torch.isnan(loss):
            pdb.set_trace()

        return 0.5 * loss + 0.5 * graph_sim

    def training_step(self, batch, batch_idx):
        (input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source,
         docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt,article_name,video_file,
         bounds,video_duration,image_feat,audio_feat,place) = batch
        loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
                                words_positions_source,
                                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
                                sents_positions_tgt,article_name,video_file,bounds,video_duration,image_feat,audio_feat,place)
        

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size_source": input_ids_source.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        return loss

    """def compute_rouge_batch(
            self,
            input_ids,
            gold_str,  
            heterograph_source,
            words_positions_source,
            sents_positions_source,
            docs_positions_source,
            batch_idx,
            article_name,
            video_file,
            bounds,
            video_duration,
            image_feat,
            audio_feat,
            place
    ):
        device = input_ids.device
      
        global_attention_mask = torch.zeros_like(input_ids).to(device)
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.docsep_token_id] = 1
        global_attention_mask[input_ids == self.sentsep_token_id] = 1

        attention_mask = torch.ones_like(input_ids).to(device)
        attention_mask[input_ids == self.pad_token_id] = 0

        
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=3 if self.args.apply_triblck else None,
            length_penalty=self.args.length_penalty,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source,
            article_name=article_name,
            video_file=video_file,
            bounds=bounds,
            video_duration=video_duration,
            image_feat=image_feat,
            audio_feat=audio_feat,
            place=place
        )

      
        generated_strs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

     
        output_dir = os.path.join(
            self.args.model_path,
            "generated_txt_%d_%s_beam=%d_%d_%d"
            % (
                self.args.mask_num,
                self.args.dataset_name,
                self.args.beam_size,
                self.args.max_length_input,
                self.args.max_length_tgt,
            ),
        )

      
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, pred in enumerate(generated_strs):
            file_name = f"{article_name}-{place}.txt" 
            with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as of:
                of.write(pred.replace("<n>", "\n"))

       
        result_batch = []
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            hyp = generated_strs[i]
            ref = gold_str[i]  # gold_str 应该是 List[str]
            scores = self.scorer.score(hyp, ref)
            
            r1 = scores["rouge1"]
            r2 = scores["rouge2"]
            rl = scores["rougeL"]
            rlsum = scores["rougeLsum"]

            r1_r, r1_p, r1_f = r1.recall, r1.precision, r1.fmeasure
            r2_r, r2_p, r2_f = r2.recall, r2.precision, r2.fmeasure
            rl_r, rl_p, rl_f = rl.recall, rl.precision, rl.fmeasure
            rlsum_r, rlsum_p, rlsum_f = rlsum.recall, rlsum.precision, rlsum.fmeasure

            # 使用 article_name 和 place 作为 ID
            result_batch.append((
                f"{article_name}-{place}",  # 将 ID 更改为 article_name-place
                r1_r, r1_p, r1_f,
                r2_r, r2_p, r2_f,
                rl_r, rl_p, rl_f,
                rlsum_r, rlsum_p, rlsum_f
            ))

        return result_batch"""

    def compute_rouge_batch(
            self,
            input_ids,
            gold_str,  # List[str]
            heterograph_source,
            words_positions_source,
            sents_positions_source,
            docs_positions_source,
            batch_idx,
            article_name,
            video_file,
            bounds,
            video_duration,
            image_feat,
            audio_feat,
            place
    ):
        device = input_ids.device
       
        global_attention_mask = torch.zeros_like(input_ids).to(device)
        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.docsep_token_id] = 1
        global_attention_mask[input_ids == self.sentsep_token_id] = 1

        attention_mask = torch.ones_like(input_ids).to(device)
        attention_mask[input_ids == self.pad_token_id] = 0

     
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.args.max_length_tgt,
            min_length=self.args.min_length_tgt,
            num_beams=self.args.beam_size,
            no_repeat_ngram_size=3 if self.args.apply_triblck else None,
            length_penalty=self.args.length_penalty,
            heterograph=heterograph_source,
            words_positions_source=words_positions_source,
            sents_positions_source=sents_positions_source,
            docs_positions_source=docs_positions_source,
            article_name=article_name,
            video_file=video_file,
            bounds=bounds,
            video_duration=video_duration,
            image_feat=image_feat,
            audio_feat=audio_feat,
            place=place
        )

        
        generated_strs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )


        if getattr(self.args, 'mode', 'train') == 'test':
            output_dir = os.path.join(
                self.args.model_path,
                f"generated_txt_{self.args.mask_num}_{self.args.dataset_name}_beam={self.args.beam_size}_{self.args.max_length_input}_{self.args.max_length_tgt}"
            )
            os.makedirs(output_dir, exist_ok=True)
            for i, pred in enumerate(generated_strs):
                art = article_name[i] if isinstance(article_name, (list, tuple)) else article_name
                plc = place[i]            if isinstance(place, (list, tuple))       else place
                file_name = f"{art}-{plc}.txt"
                with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as of:
                    of.write(pred.replace('<n>', '\n'))

      
        result_batch = []
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            hyp = generated_strs[i]
            ref = gold_str[i]
            scores = self.scorer.score(hyp, ref)
            r1 = scores['rouge1']
            r2 = scores['rouge2']
            rl = scores['rougeL']
            rlsum = scores['rougeLsum']
            result_batch.append((
                f"{article_name[i] if isinstance(article_name, (list, tuple)) else article_name}-"
                f"{place[i] if isinstance(place, (list, tuple)) else place}",
                r1.recall, r1.precision, r1.fmeasure,
                r2.recall, r2.precision, r2.fmeasure,
                rl.recall, rl.precision, rl.fmeasure,
                rlsum.recall, rlsum.precision, rlsum.fmeasure
            ))

        return result_batch


    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        input_ids_source, output_ids, input_ids_summary, heterograph_source, words_positions_source, sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt, sents_positions_tgt, tgt, article_name, video_file, bounds, video_duration, image_feat, audio_feat, place = batch
        loss = self.shared_step(input_ids_source, output_ids, input_ids_summary, heterograph_source,
                                words_positions_source,
                                sents_positions_source, docs_positions_source, heterograph_tgt, words_positions_tgt,
                                sents_positions_tgt, article_name, video_file, bounds, video_duration, image_feat,
                                audio_feat, place)
        # print("validation step", input_ids_source.size())
        if self.args.compute_rouge:
            result_batch = self.compute_rouge_batch(input_ids_source, tgt, heterograph_source, words_positions_source,
                                                    sents_positions_source, docs_positions_source, batch_idx,
                                                    article_name, video_file, bounds, video_duration, image_feat,
                                                    audio_feat, place)
            return {"vloss": loss, "rouge_result": result_batch}
        else:
            return {"vloss": loss}

    def compute_rouge_all(self, outputs, output_file=None):
       
        rouge_result_all = [r for batch_out in outputs for r in batch_out["rouge_result"]]

        
        ids = [row[0] for row in rouge_result_all]  # ['article1-Intro', 'article1-Method', ...]
        metrics_data = [row[1:] for row in rouge_result_all]  # [[r1_r, r1_p, ..., rlsum_f], [...], ...]

        
        metric_cols = [
            "rouge-1-r", "rouge-1-p", "rouge-1-f",
            "rouge-2-r", "rouge-2-p", "rouge-2-f",
            "rouge-L-r", "rouge-L-p", "rouge-L-f",
            "rouge-Lsum-r", "rouge-Lsum-p", "rouge-Lsum-f"
        ]

       
        rouge_results = pd.DataFrame(metrics_data, columns=metric_cols, index=ids)

        
        avg_series = rouge_results.mean(axis=0)  

        
        rouge_results.loc["avg_score"] = avg_series

       
        if output_file:
            csv_name = (
                    self.args.model_path
                    + output_file
                    + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
            )
          
            rouge_results.to_csv(csv_name)

      
        avg_list = avg_series.tolist()  # 把 pandas Series 转成 Python 列表：[avg_r1_r, avg_r1_p, ..., avg_rlsum_f]

       
        avgf = (avg_list[2] + avg_list[5] + avg_list[11]) / 3
        metrics = avg_list

        print(f"Validation Result at Step {self.global_step}")
        print(
            "Rouge-1 r: %.6f, p: %.6f, f: %.6f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r: %.6f, p: %.6f, f: %.6f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r: %.6f, p: %.6f, f: %.6f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r: %.6f, p: %.6f, f: %.6f"
            % (metrics[9], metrics[10], metrics[11])
        )

       
        names = metric_cols
        return names, metrics, avgf

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_rouge:
            names, metrics, avgf = self.compute_rouge_all(outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log(
                "avgf", avgf,
                on_step=False,  
                on_epoch=True,  
                prog_bar=True,  
                logger=True  
            )

            return {
                "avg_val_loss": vloss,
                "avgf": avgf,
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        tloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
        output_file = "test_%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        names, metrics, avgf = self.compute_rouge_all(outputs, output_file=output_file)
        metrics = [tloss, avgf] + metrics
        names = ["tloss", "avgf"] + names
        logs = dict(zip(*[names, metrics]))
        self.logger.log_metrics(logs, step=self.global_step)
        self.log("avgf", avgf)
        # self.log_dict(logs)
        return {"avg_test_loss": tloss, "avgf": avgf, "log": logs, "progress_bar": logs}


def train(args):
    model = HGSummarizer(args)

    # load dataset
    train_dataloader = get_dataloader_summ(args, model.tokenizer, 'train', args.num_workers, True)
    valid_dataloader = get_dataloader_summ(args, model.tokenizer, 'validation', args.num_workers, False)
    test_dataloader = get_dataloader_summ(args, model.tokenizer, 'test', args.num_workers, False)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(args.model_path, "summ_checkpoints/")

    """checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}",
        save_top_k=args.save_top_k,
        monitor="vloss",
        mode="min",
        save_on_train_epoch_end=False,
    )
   
    early_stopping = EarlyStopping(
    monitor="vloss",       
    patience=args.early_stop_patience,
    mode="min",          
    verbose=True,        
    )"""

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-avgf={avgf:.4f}",
        save_top_k=args.save_top_k,
        monitor="avgf",
        mode="max",
        save_on_train_epoch_end=False,
        save_last=True,
    )
  
    early_stopping = EarlyStopping(
        monitor="avgf",  
        patience=args.early_stop_patience,
        mode="max",  
        verbose=True,  
    )
    #early_stopping = EarlyStopping(monitor='vloss', patience=3, mode='min')

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name=args.model_name)

    # initialize trainer
    """trainer = pl.Trainer(
        devices=args.gpus,
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        #auto_select_gpus=True,
        strategy=args.speed_strategy,
        #track_grad_norm=-1,
        #max_steps=args.total_steps * args.accum_batch,
        #replace_sampler_ddp=False,
        accumulate_grad_batches=args.accum_batch,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stopping],
        precision=16,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches else 1.0,
        num_sanity_val_steps=0
    )"""
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator=args.accelerator,
        auto_select_gpus=True,
        strategy=args.speed_strategy,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.accum_batch,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 1,
        logger=logger,
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        precision=16,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches else 1.0,
        num_sanity_val_steps=0
    )

    if args.resume_ckpt is not None:
        model = HGSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = HGSummarizer(args)

    # pdb.set_trace()
    trainer.fit(model, train_dataloader, test_dataloader)

    #model.enable_input_require_grads()

    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    # initialize trainer
    """trainer = pl.Trainer(
        devices=1,
        auto_select_gpus=True,
        accelerator=args.accelerator,
        track_grad_norm=-1,
        max_epochs=args.max_epochs,
        #max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=5,
        precision=16,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0
    )"""
    trainer = pl.Trainer(
        devices=1,
        auto_select_gpus=True,
        accelerator=args.accelerator,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.accum_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=1,
        precision=16,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0
    )

    if args.resume_ckpt is not None:
        model = HGSummarizer.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = HGSummarizer(args)

    # load dataset
    test_dataloader = get_dataloader_summ(args, model.tokenizer, 'test', args.num_workers, False)

    # test
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    parser = argparse.ArgumentParser()

    # Gneral
    parser.add_argument("--gpus", default=1, type=int, help="The number of gpus to use")
    parser.add_argument("--accelerator", default="gpu", type=str, choices=["gpu", "cpu"])
    parser.add_argument("--speed_strategy", default=None, type=str, help="Accelerator strategy, e.g., ddp")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--model_name", default="HGSum")
    parser.add_argument("--pretrained_primer", type=str, default=None,
                        help="Name or path of pretrained PRIMERA from Huggingface, or the model to be tested")
    parser.add_argument("--with_sent_sep", action="store_true",
                        help="Insert <sent-sep> at the end of each sentence when concatenating different documents")
    parser.add_argument("--debug_mode", action="store_true", help="Set true if to debug")
    parser.add_argument("--compute_rouge", action="store_true", help="whether to compute rouge in validation steps")
    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--model_path", type=str, default="result/",
                        help="The path to save output and checkpoints in training and testing")
    parser.add_argument("--ckpt_path", type=str, default=None, help="dir to save checkpoints")
    parser.add_argument("--save_top_k", default=5, type=int)
    parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from", default=None)
    parser.add_argument("--data_path", type=str, default="../../datasets/")
    parser.add_argument("--dataset_name", type=str, default="multinews",
                        choices=["multinews", "arxiv", "multixscience", "wcep_10", "wcep_100", "peersum_r",
                                 "peersum_rc", "peersum_all","other"])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for dataloader")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument("--adafactor", action="store_true", help="Use adafactor optimizer")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--rand_seed", type=int, default=42,
                        help="Seed for random sampling, useful for few shot learning")

    # For training
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Use limited batches in training")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Use limited batches in validation")
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--accum_data_per_step", type=int, default=16, help="Number of data per step")
    parser.add_argument("--total_steps", type=int, default=50000, help="Number of steps to train")
    parser.add_argument("--num_train_data", type=int, default=-1,
                        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use")
    parser.add_argument("--fix_lr", action="store_true", help="use fix learning rate")
    parser.add_argument("--test_imediate", action="store_true", help="test on the best checkpoint")
    parser.add_argument("--fewshot", action="store_true", help="whether this is a run for few shot learning")

    # For testing
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="Number of batches to test in the test mode")
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument("--length_penalty", type=float, default=1, help="length penalty of generated text")
    parser.add_argument("--mask_num", type=int, default=0, help="Number of masks in the input of summarization data")
    parser.add_argument("--test_batch_size", type=int, default=-1,
                        help="batch size for test, used in few shot evaluation.")
    parser.add_argument("--apply_triblck", action="store_true",
                        help="whether apply trigram block in the evaluation phase")
    parser.add_argument("--num_test_data", type=int, default=-1, help="the number of testing data")
    #add new arg
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--no_compute_rouge", action="store_true", help="disable rouge")
    parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=5,
    help="early stop")


    args = parser.parse_args()
    if args.no_compute_rouge:
        args.compute_rouge = False
    args.accum_batch = args.accum_data_per_step // args.batch_size

    if args.gpus > 0:
        args.accelerator = "gpu"
    else:
        args.accelerator = "cpu"
        args.gpus = 1
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    print(args)

    if not os.path.exists(args.model_path):  # this is used to save the checkpoints and logs
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    if args.mode == "test":
        test(args)
#datamultinews_graph_noun_sentem
