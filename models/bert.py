import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from transformers import RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import logging

class BertFNModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, dataset):
        super(BertFNModel, self).__init__()
        if dataset == 'ch':
            self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)

        self.mlp = MLP(emb_dim, mlp_dims, dropout)
        self.attention = MaskAttention(emb_dim)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        bert_feature = self.bert(inputs, attention_mask = masks).last_hidden_state
        bert_feature, _ = self.attention(bert_feature, masks)
        output = self.mlp(bert_feature)
        return torch.sigmoid(output.squeeze(1))


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 dataset,
                 early_stop = 5,
                 epoches = 100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict

        # CHANGED: Haz que use_cuda respete la disponibilidad real de CUDA para evitar el crash.
        # Si el usuario pide CUDA pero no hay GPU, se cae a CPU sin reventar.
        self.use_cuda = bool(use_cuda) and torch.cuda.is_available()  # CHANGED
        self.device = torch.device("cuda") if self.use_cuda else torch.device("cpu")  # CHANGED

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.dataset = dataset
        
        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir)
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
            # CHANGED: Mensaje claro del device final elegido.
            logger.info(f'Using device: {self.device.type}')  # CHANGED

        self.model = BertFNModel(self.emb_dim, self.mlp_dims, self.dropout, self.dataset)
        # CHANGED: Evita .cuda() directo. Usa .to(device) para soportar CPU/GPU sin romper.
        self.model = self.model.to(self.device)  # CHANGED

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        best_metric = recorder.cur['metric']
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                # CHANGED: data2gpu ya mueve tensores si self.use_cuda=True; en CPU no los mueve.
                batch_data = data2gpu(batch, self.use_cuda)  # CHANGED (comentario)

                label = batch_data['label']

                # CHANGED: Había un doble optimizer.zero_grad(); deja solo uno.
                optimizer.zero_grad()  # CHANGED
                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; batch_loss = {2}; average_loss = {3}'.format(epoch, str(self.lr), loss.item(), avg_loss)

            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_param_dir, 'parameter_bert.pkl'))
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue

        # CHANGED: Carga el checkpoint respetando el device para evitar error en CPU.
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_bert.pkl'),
                                              map_location=self.device))  # CHANGED
        results = self.test(self.test_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_bert.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                # CHANGED: Igual que en train, deja que data2gpu maneje el traslado condicional.
                batch_data = data2gpu(batch, self.use_cuda)  # CHANGED (comentario)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)
