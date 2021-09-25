import argparse, json, os, random, math
import torch
from transformers import LongformerForMaskedLM, LongformerTokenizer
from torch.utils.data import Dataset, DataLoader

def get_entities_mask(entities_list, max_seq_length=1024, mask_size = 0.5):
    entities_masks = []
    for entities in entities_list:
        #randomly select subset of entities
        entities_subset = random.sample(entities, math.ceil(mask_size*len(entities)))
        #create a mask of the positions of these 0.1 entities relative to max length
        doc_entity_mask = torch.zeros(max_seq_length)
        for start, end in entities_subset:
            # +1 to cater for <CLS>
            start, end = start+1, end+1
            doc_entity_mask[start:end]=1
        entities_masks.append(doc_entity_mask)
    entities_masks = torch.stack(entities_masks)
    return entities_masks.gt(0)

class DWIE_Data(Dataset):
    def __init__(self, documents, entities, tokenizer):
        self.documents = documents
        self.entities = entities
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 4096
        self.encodings = self.tokenizer(self.documents, padding=True, truncation=True, return_tensors="pt")
        self.labels = self.encodings["input_ids"]
        self.encodings["input_ids"]=self.mask_entities(entities, self.encodings["input_ids"])

    def mask_entities(self, entities, input_ids):
        entities_mask  = get_entities_mask(entities, max_seq_length=input_ids.size()[-1])
        # Get input ids and replace with mask token
        input_ids[entities_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return input_ids

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def collate_fn(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch]) 
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])
        labels = torch.stack([ex['labels'] for ex in batch]) 

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# tokenizer.model_max_length=1024
# train = dwie["train"]
# train_sentences = [' '.join(doc["sentences"]) for doc in train]
# train_entities = [[(entity["start"], entity["end"]) for entity in doc["entities"]] for doc in train]
# input_ids = tokenizer(train_sentences, padding=True, truncation=True, return_tensors="pt")["input_ids"]
# entities_mask  = get_entities_mask(train_entities, input_ids.size()[-1], mask_size = 0.3)
# original_input_ids = input_ids.clone()
# input_ids[entities_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
# sample1 = tokenizer.decode(original_input_ids[0])
# sample2 = tokenizer.decode(input_ids[0])
# import ipdb; ipdb.set_trace()
# import sys; sys.exit()

import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class NERLongformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        #self.config =  
        self.longformer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True, output_hidden_states=True)
        self.softmax = nn.Softmax(dim=-1)
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


    def val_dataloader(self, dwie):
        val = dwie["test"]
        val_sentences = [' '.join(doc["sentences"]) for doc in val]
        val_entities = [[(entity["start"], entity["end"]) for entity in doc["entities"]] for doc in val]
        val_data = DWIE_Data(val_sentences, val_entities, self.tokenizer)
        val_dataloader = DataLoader(val_data, batch_size=self.args.eval_batch_size, collate_fn = val_data.collate_fn)
        return val_dataloader

    def train_dataloader(self, dwie):
        train = dwie["train"]
        train_sentences = [' '.join(doc["sentences"]) for doc in train]
        train_entities = [[(entity["start"], entity["end"]) for entity in doc["entities"]] for doc in train]
        train_data = DWIE_Data(train_sentences, train_entities, self.tokenizer)
        train_dataloader = DataLoader(train_data, batch_size=self.args.train_batch_size, collate_fn = train_data.collate_fn)
        return train_dataloader

    def forward(self, **batch):
        # in lightning, forward defines the prediction/inference actions
        #input_ids, attention_mask, labels = batch
        outputs = self.longformer(**batch, return_dict=False)
        return outputs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # logits = torch.argmax(self.softmax(logits), dim=-1)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        #input_ids, attention_mask, labels = batch
        loss, logits = self(**batch)
        # print(hidden_states)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        logs = {
            "val_loss": val_loss_mean,
            "val_perplexity": torch.exp(val_loss_mean)
        }

        # clearlogger.report_scalar(title='perplexity', series = 'val_perplexity', value=logs["val_perplexity"], iteration=self.trainer.current_epoch) 

        self.log("val_loss", logs["val_loss"])
        self.log("val_perplexity", logs["val_perplexity"])

    # Freeze weights?
    def configure_optimizers(self):

        # Freeze 1st 6 layers of longformer
        for idx, (name, parameters) in enumerate(self.longformer.named_parameters()):
            if idx<6:
                parameters.requires_grad=False
            else:
                parameters.requires_grad=True

        optimizer = torch.optim.Adam(self.longformer.parameters(), lr=1e-4)
        return optimizer

