from clearml import Task, StorageManager, Dataset
import argparse
import json

task = Task.init(project_name='longGRIT', task_name='LEDGRIT', output_uri="s3://experiment-logging/storage/")
clearlogger = task.get_logger()

#task.set_base_docker("nvcr.io/nvidia/pytorch:21.05-py3")
#task.set_base_docker("default-base")

config = json.load(open('config.json'))
args = argparse.Namespace(**config)

task.connect(args)
task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.execute_remotely(queue_name="128RAMv100", exit_process=True)

import glob
import logging
import os
from collections import OrderedDict
from eval import eval_ceaf

import numpy as np
from torch import nn
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.nn import CrossEntropyLoss, Softmax, Linear, NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from transformer_base import BaseTransformer, add_generic_args, generic_train
from utils_s_t import convert_examples_to_features, get_labels, read_examples_from_file, read_golds_from_test_file, not_sub_string, remove_led_prefix_from_tokens

role_list = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NERTransformer(BaseTransformer):
    """
    A training module for single-transformer-ee. See BaseTransformer for the core options.
    """

    mode = "base"

    def __init__(self, hparams):
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        # super(NERTransformer, self).__init__(hparams, num_labels, self.mode)
        super(NERTransformer, self).__init__(hparams, self.mode)

        # n_gpu = torch.cuda.device_count()
        # self.MASK = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.SEP = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self.CLS = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.linear = Linear(768, self.hparams.max_seq_length_src)

    def forward(self, **inputs):

        #print(', '.join(['{}={!r}'.format(k, v.size()) for k, v in inputs.items()]))
        decoder_input_mask_1d = inputs.pop("decoder_attention_mask", None)
        labels = inputs.pop("labels", None) # doc_length

        batch_size = inputs['input_ids'].size()[0]
        global_attention_mask_cls = (inputs['input_ids']==self.CLS).type(torch.uint8)
        global_attention_mask_sep = (inputs['input_ids']==self.SEP).type(torch.uint8)
        global_attention_mask_period = (inputs['input_ids']==self.tokenizer.convert_tokens_to_ids('.')).type(torch.uint8)
        global_attention_mask = global_attention_mask_cls + global_attention_mask_sep + global_attention_mask_period

        #input_ids = inputs.pop("input_ids", None)
        #input_embeds = self.longformer(input_ids=input_ids)
        #inputs = {**inputs, "global_attention_mask": global_attention_mask, "output_attentions":True, "inputs_embeds": input_embeds[1][-1]}
        inputs = {**inputs, "global_attention_mask": global_attention_mask, "output_attentions":True}
        outputs = self.model(**inputs) # sequence_output, pooled_output, (hidden_states), (attentions)       

        encoder_last_hidden_state = outputs.encoder_last_hidden_state # seq length by hidden size
        encoder_last_hidden_state = torch.transpose(encoder_last_hidden_state, 1, 2)
        decoder_last_hidden_state = outputs.last_hidden_state # tgt length by hidden size
        logits = torch.bmm(decoder_last_hidden_state, encoder_last_hidden_state) #tgt length by seq length        

        loss = None
        outputs=(logits,)

        if labels!=None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if decoder_input_mask_1d is not None:
                active_loss = decoder_input_mask_1d.view(-1) == 1 # Convert to a single dimension where True if equals 1 and 0 if not
                active_logits = logits.view(-1, args.max_seq_length_src) # Collapse batch to single dimension 
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                ) # if is in active loss, collapse batch of labels to single dimension else replace with the ignore ignore index from loss function
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, args.max_seq_length_src), labels.view(-1))

            outputs = (loss,) + (logits,)
        # import ipdb; ipdb.set_trace()
        return outputs

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "decoder_input_ids": batch[2], "decoder_attention_mask": batch[3], "labels": batch[4], "position_ids": batch[5]}

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "decoder_input_ids": batch[2], "decoder_attention_mask": batch[3], "labels": batch[4]}
        
        outputs = self(**inputs)
        loss = outputs[0]

        return {"loss": loss}

    def prepare_data(self):
        "Called to initialize data. Use the call to construct features"
        args = self.hparams
        for mode in ["train", "dev", "test"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file):
                logger.info("Creating features from dataset file at %s", args.data_dir)
                examples = read_examples_from_file(args.data_dir, mode, self.tokenizer)

                features = convert_examples_to_features(
                    examples,
                    # self.labels,
                    args.max_seq_length_src,
                    args.max_seq_length_tgt,
                    self.tokenizer,
                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=bool(args.model_type in ["roberta"]),
                    pad_on_left=bool(args.model_type in ["xlnet"]),
                    pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                    pad_token_label_id=self.pad_token_label_id,
                )

                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    def load_dataset(self, mode, batch_size):
        "Load datasets. Called after prepare data."
        args = self.hparams
        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if args.debug:
            features = features[:5]
            # features = features[:len(features)//10]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_decoder_input_ids = torch.tensor([f.decoder_input_ids for f in features], dtype=torch.long)
        all_decoder_input_mask = torch.tensor([f.decoder_input_mask for f in features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_docid = torch.tensor([f.docid for f in features], dtype=torch.long)
        return DataLoader(
            TensorDataset(all_input_ids, all_input_mask, all_decoder_input_ids, all_decoder_input_mask, all_label_ids, all_position_ids, all_docid), num_workers=4, batch_size=batch_size
        )

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "decoder_input_ids": batch[2], "decoder_attention_mask": batch[3], "labels": batch[4], "position_ids": batch[5]}
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "decoder_input_ids": batch[2], "decoder_attention_mask": batch[3], "labels": batch[4]}
        
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        docid = batch[5].detach().cpu().numpy()        

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids, "input_ids": inputs["input_ids"].detach().cpu().numpy(), "docid": docid}


    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]): # For row in logits
            for j in range(out_label_ids.shape[1]): # For column in logits
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(out_label_ids[i][j])
                    preds_list[i].append(preds[i][j])
        # import ipdb; ipdb.set_trace()

        acc = accuracy_score(out_label_list, preds_list)

        logs = {
            "val_loss": val_loss_mean.item(),
            "val_accuracy": acc.item()
        }

        clearlogger.report_scalar(title='accuracy', series = 'val_acc', value=acc, iteration=self.trainer.current_epoch) 

        self.log("val_loss", logs["val_loss"])
        self.log("val_accuracy", logs["val_accuracy"])


    def test_step(self, batch, batch_nb):
        "Compute test"
        # test_loss
        tgt_input_ids, init_tgt_input_ids, tgt_input_mask = torch.tensor([self.CLS]), torch.tensor([self.CLS]), torch.tensor([1])        
        tgt_position_ids, init_tgt_position_ids = torch.tensor([0]), torch.tensor([0])

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "decoder_input_ids": tgt_input_ids, "labels": batch[4]}
        
        bs = batch[0].size()[0]
        tgt_input_ids, init_tgt_input_ids, tgt_input_mask = np.repeat(tgt_input_ids.unsqueeze(0), bs, 0).to(device), np.repeat(init_tgt_input_ids.unsqueeze(0), bs, 0).to(device), np.repeat(tgt_input_mask.unsqueeze(0), bs, 0).to(device)        
        tgt_position_ids, init_tgt_position_ids = np.repeat(tgt_position_ids.unsqueeze(0), bs, 0).to(device), np.repeat(init_tgt_position_ids.unsqueeze(0), bs, 0).to(device)        

        i = 0
        while i < self.hparams.max_seq_length_tgt:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]           
            tgt_input_mask = torch.cat((tgt_input_mask, torch.ones([bs, len(tgt_input_ids)]).to(device)), dim=1)

            params = {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": tgt_input_ids, 'decoder_attention_mask': tgt_input_mask}
            outputs = self(**params)
            logits = outputs[0][0] #Only get the 1st row of the logits

            # (constraint 2) thresh for predicting [SEP] # if top 1 is SEP replace it with the 2nd highest token
            probs = torch.nn.Softmax(dim=-1)(logits)
            top_2_probs, top_2_indices = torch.topk(probs, 2, dim=-1)
            for j in range(top_2_indices.size(0)):
                prob_gap = (top_2_probs[j][0]/top_2_probs[j][1]).detach().cpu().tolist()
                if input_ids[0][top_2_indices[j][0].detach().cpu().tolist()].detach().cpu().tolist() == self.SEP and prob_gap < global_args.thresh:
                    top_2_indices[j][0] = top_2_indices[j][1]

            # only get the 1st column(top 1) of all rows in top2 indices
            out_position_id = top_2_indices[:, 0]

            ## option 2: direct greedy decoding
            #out_position_id = torch.argmax(logits, -1)
            
            # print(out_position_id) # debug
            out_input_id = torch.index_select(input_ids, 1, out_position_id)
            out_position_id = out_position_id.unsqueeze(dim=0) # add batch dim
            tgt_input_ids = torch.cat((init_tgt_input_ids, out_input_id), dim=1).squeeze(-1)
            #import ipdb; ipdb.set_trace()
            tgt_position_ids = torch.cat((init_tgt_position_ids, out_position_id), dim=1)
            i+=1

        # from out_input_id_list (pred_seq) to pred_extracts
        docids = batch[6].detach().cpu().tolist()
        pred_seq = []
        pred_extract = []
        for b in range(bs):
            src_input_id_list = input_ids[b].detach().cpu().tolist()
            out_input_id_list = out_input_id[b].detach().cpu().tolist()
            out_position_id_list = out_position_id[b].detach().cpu().tolist()
            if out_input_id_list[-1] != self.SEP:
                out_input_id_list.append(self.SEP)

            # get raw pred_seq
            sep_cnt = 0
            for idx, token_id in enumerate(out_input_id_list):
                if token_id == self.SEP:
                    sep_cnt += 1
                if sep_cnt >= 5: break
            
            seq = self.tokenizer.convert_ids_to_tokens(out_input_id_list[:idx+1])
            pred_seq.append(remove_led_prefix_from_tokens(seq))

            # get pred_extract
            p_extract = []
            sep_cnt = 0
            position_buf = []
            for idx, token_id in enumerate(out_input_id_list):
                if token_id == self.SEP:
                    sep_cnt += 1
                    entitys = []
                    s_e_pair = []
                    for position in position_buf:
                        s_e_pair.append(position)
                        if len(s_e_pair) == 2:
                            s, e = s_e_pair[0], s_e_pair[1]
                            extract_ids = []
                            for j in range(s, e+1): 
                                extract_ids.append(src_input_id_list[j])
                            extract_tokens = remove_led_prefix_from_tokens(self.tokenizer.convert_ids_to_tokens(extract_ids))
                            if extract_tokens:
                                if len(extract_tokens) <= 20: 
                                    candidate_str = " ".join(extract_tokens).replace(" ##", "")
                                    if sep_cnt != 4 or "bomb" not in candidate_str:
                                        if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
                                            entitys.append([candidate_str])
                            s_e_pair = []
                    
                    # extra s in s_e_pair
                    if s_e_pair:
                        extract_tokens = remove_led_prefix_from_tokens(self.tokenizer.convert_ids_to_tokens([src_input_id_list[s_e_pair[0]]]))
                        if len(extract_tokens) <= 20: 
                            candidate_str = " ".join(extract_tokens).replace(" ##", "")
                            if sep_cnt != 4 or "bomb" not in candidate_str:
                                if [candidate_str] not in entitys and not_sub_string(candidate_str, entitys) and candidate_str[:2] != "##":
                                    entitys.append([candidate_str])

                    # add all entitys of this role
                    p_extract.append(entitys)
                    # clean buffer
                    position_buf = []
                else:
                    position_buf.append(out_position_id_list[idx])

                if sep_cnt >= 5: break
            
            # import ipdb; ipdb.set_trace()
            pred_extract.append(p_extract)

        return {"pred_seq": pred_seq, "pred_extract": pred_extract, "docid": docids, "labels": batch[4].detach().cpu().tolist(), "input_ids": inputs["input_ids"].detach().cpu().numpy()}


    def test_epoch_end(self, outputs):
        # # updating to test_epoch_end instead of deprecated test_end
        args = self.hparams
        logs = {}

        ## real decoding
        # read golds
        doctexts_tokens, golds = read_golds_from_test_file(args.data_dir, self.tokenizer)
        #import ipdb; ipdb.set_trace()

        if args.debug:
            doctexts_tokens, golds = {key: doctexts_tokens[key] for key in list(doctexts_tokens.keys())[:5]}, {key: golds[key] for key in list(golds.keys())[:5]}
        # get preds and preds_log
        preds = OrderedDict()
        preds_log = OrderedDict()
        for x in outputs:
            docids = x["docid"]
            pred_seq = x["pred_seq"]
            pred_extract = x["pred_extract"]
            labels = x["labels"]
            converted = x["input_ids"]
            converted = np.concatenate([[sent[label] for label in labels] for sent in converted], axis=0)
            converted = [remove_led_prefix_from_tokens(self.tokenizer.convert_ids_to_tokens(sent)) for sent in converted]
            converted = [[token for token in sent if token!="<pad>"] for sent in converted]

            # preds (pred_extract)
            for docid, p_extract in zip(docids, pred_extract):
                if docid not in preds:
                    preds[docid] = OrderedDict()
                    for idx, role in enumerate(role_list):
                        preds[docid][role] = []
                        if idx+1 > len(p_extract): 
                            continue
                        elif p_extract[idx]:
                            preds[docid][role] = [pred for pred in p_extract[idx]]
                            
            # preds_log
            for docid, p_seq in zip(docids, pred_seq):
                if docid not in preds_log:
                    preds_log[docid] = OrderedDict()
                    preds_log[docid]["doctext"] = " ".join([text.replace("\u0120", "") for text in doctexts_tokens[docid]])
                    preds_log[docid]["pred_seq"] = " ".join(p_seq)
                    preds_log[docid]["pred_extracts"] = preds[docid]
                    preds_log[docid]["gold_extracts"] = golds[docid]
                    preds_log[docid]["labels"] = converted

                    # clearlogger.report_text(preds_log[docid]["doctext"], level=logging.DEBUG, print_console=False)
                    # clearlogger.report_text(preds_log[docid]["pred_seq"], level=logging.DEBUG, print_console=False)
                    # clearlogger.report_text(preds_log[docid]["pred_extracts"], level=logging.DEBUG, print_console=False)
                    # clearlogger.report_text(preds_log[docid]["gold_extracts"], level=logging.DEBUG, print_console=False)
                    # clearlogger.report_text(preds_log[docid]["labels"], level=logging.DEBUG, print_console=False)


        # evaluate
        results = eval_ceaf(preds, golds)
        logger.info("================= CEAF score =================")
        logger.info("phi_strict: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["strict"]["micro_avg"]["p"] * 100, results["strict"]["micro_avg"]["r"] * 100, results["strict"]["micro_avg"]["f1"] * 100))
        logger.info("phi_prop: P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results["prop"]["micro_avg"]["p"] * 100, results["prop"]["micro_avg"]["r"] * 100, results["prop"]["micro_avg"]["f1"] * 100))
        logger.info("==============================================")
        logs["test_micro_avg_f1_phi_strict"] = results["strict"]["micro_avg"]["f1"]
        logs["test_micro_avg_precision_phi_strict"] = results["strict"]["micro_avg"]["p"]
        logs["test_micro_avg_recall_phi_strict"] = results["strict"]["micro_avg"]["r"]

        clearlogger.report_scalar(title='f1', series = 'test', value=logs["test_micro_avg_f1_phi_strict"], iteration=1) 
        clearlogger.report_scalar(title='precision', series = 'test', value=logs["test_micro_avg_precision_phi_strict"], iteration=1) 
        clearlogger.report_scalar(title='recall', series = 'test', value=logs["test_micro_avg_recall_phi_strict"], iteration=1) 

        logger.info("writing preds to .out file:")


        # import ipdb; ipdb.set_trace()
        return {"log": logs, "progress_bar": logs, "preds": preds_log}


    def configure_optimizers(self):
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
            parameters.requires_grad=False
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt=optimizer
        return optimizer


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length_src",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization for src. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--max_seq_length_tgt",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization for tgt. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--labels",
            default="",
            type=str,
            help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        parser.add_argument("--debug", action="store_true", help="if in debug mode")

        parser.add_argument("--thresh", default=1, type=float, help="thresh for predicting [SEP]",)
        return parser


#parser = argparse.ArgumentParser()
#add_generic_args(parser, os.getcwd())
#parser = NERTransformer.add_model_specific_args(parser, os.getcwd())
#args = parser.parse_args()
### To create config.json
##args_dict = vars(args)
##json.dump(args_dict, open('config.json', 'w'))


#Read args from config file instead, use vars() to convert namespace to dict
# dataset = Dataset.get(dataset_name="wikievents-muc4", dataset_project="datasets/wikievents", dataset_tags=["muc4-format"], only_published=True)
# dataset_folder = dataset.get_local_copy()
# # if os.path.exists(dataset_folder)==False:
# os.symlink(os.path.join(dataset_folder, "data/wikievents/muc_format"), args.data_dir)

class bucket_ops:
    StorageManager.set_cache_file_limit(5, cache_context=None)

    def list(remote_path:str):
        return StorageManager.list(remote_path, return_full_path=False)

    def upload_folder(local_path:str, remote_path:str):
        StorageManager.upload_folder(local_path, remote_path, match_wildcard=None)
        print("Uploaded {}".format(local_path))

    def download_folder(local_path:str, remote_path:str):
        StorageManager.download_folder(remote_path, local_path, match_wildcard=None, overwrite=True)
        print("Downloaded {}".format(remote_path))
    
    def get_file(remote_path:str):        
        object = StorageManager.get_local_copy(remote_path)
        return object

    def upload_file(local_path:str, remote_path:str):
        StorageManager.upload_file(local_path, remote_path, wait_for_upload=True, retries=3)


global_args = args
logger.info(args)
model = NERTransformer(args)
trainer = generic_train(model, args)

results = trainer.test(model)
task.close()


# model_name = 'longformer-base-4096'#'allenai/led-base-16384'
# save_dir = 'longformer-base-4096'
# config = AutoConfig.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# config.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)
