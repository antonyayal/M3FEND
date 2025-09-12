import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.metrics import *
from sklearn.cluster import KMeans
from transformers import BertModel, RobertaModel

from .layers import *
from utils.utils import data2gpu, Averager, metrics, Recorder


def cal_length(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1))


def norm(x: torch.Tensor) -> torch.Tensor:
    length = cal_length(x).view(-1, 1).clamp(min=1e-12)
    return x / length


def convert_to_onehot(label: torch.Tensor, batch_size: int, num: int, device: torch.device) -> torch.Tensor:
    # label must be LongTensor of shape (N, 1) or (N,)
    label = label.view(-1, 1).long()
    return torch.zeros(batch_size, num, device=device).scatter_(1, label, 1)


class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10, device: torch.device = torch.device("cpu")):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_memory = dict()
        self.device = device

    def forward(self, feature: torch.Tensor, category: torch.Tensor):
        feature = norm(feature)
        # Build list of memories for each domain
        domain_memory = []
        for i in range(self.domain_num):
            domain_memory.append(self.domain_memory[i])  # [memory_num, emb_dim], on device

        sep_domain_embedding = []
        for i in range(self.domain_num):
            # [B, emb] x [emb, M] -> [B, M]
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1
            )
            # [B, M] x [M, emb] -> [B, emb]
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        # [B, D, emb]
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        # [B, D, emb] bmm [B, emb, 1] -> [B, D, 1] -> [B, D]
        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()
        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)
        return domain_att  # [B, 1, D]

    def write(self, all_feature: torch.Tensor, category: torch.Tensor):
        # category: [B]
        domain_fea_dict = {}
        domain_set = set(category.detach().cpu().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)  # [Ni, emb_all]
            topic_att = torch.nn.functional.softmax(
                torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1
            ).unsqueeze(2)  # [Ni, M, 1]
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)  # [Ni, M, emb_all]
            new_mem = tmp_fea * topic_att  # [Ni, M, emb_all]
            new_mem = new_mem.mean(dim=0)  # [M, emb_all]
            topic_att = torch.mean(topic_att, 0).view(-1, 1)  # [M, 1]
            # EMA-like update
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem


class M3FENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, semantic_num, emotion_num, style_num, LNN_dim, domain_num, dataset):
        super(M3FENDModel, self).__init__()
        self.domain_num = domain_num
        self.dataset = dataset
        self.gamma = 10
        self.memory_num = 10
        self.semantic_num_expert = semantic_num
        self.emotion_num_expert = emotion_num
        self.style_num_expert = style_num
        self.LNN_dim = LNN_dim
        self.fea_size = 256
        self.emb_dim = emb_dim

        # Decide device dynamically; module .to(device) / .cuda() in Trainer will move parameters later
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(
            'semantic_num_expert:', self.semantic_num_expert,
            'emotion_num_expert:', self.emotion_num_expert,
            'style_num_expert:', self.style_num_expert,
            'lnn_dim:', self.LNN_dim
        )

        # Backbone encoder depending on dataset language
        if self.dataset == 'ch':
            self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        elif self.dataset == 'en':
            self.bert = RobertaModel.from_pretrained('roberta-base').requires_grad_(False)
        else:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Expected 'ch' or 'en'.")

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        # Content experts (CNN over token features)
        content_expert = []
        for _ in range(self.semantic_num_expert):
            content_expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.content_expert = nn.ModuleList(content_expert)

        # Emotion experts
        emotion_expert = []
        for _ in range(self.emotion_num_expert):
            if self.dataset == 'ch':
                emotion_expert.append(MLP(47 * 5, [256, 320], dropout, output_layer=False))
            elif self.dataset == 'en':
                emotion_expert.append(MLP(38 * 5, [256, 320], dropout, output_layer=False))
        self.emotion_expert = nn.ModuleList(emotion_expert)

        # Style experts
        style_expert = []
        for _ in range(self.style_num_expert):
            if self.dataset == 'ch':
                style_expert.append(MLP(48, [256, 320], dropout, output_layer=False))
            elif self.dataset == 'en':
                style_expert.append(MLP(32, [256, 320], dropout, output_layer=False))
        self.style_expert = nn.ModuleList(style_expert)

        # Gate over experts
        self.gate = nn.Sequential(
            nn.Linear(self.emb_dim * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[-1], self.LNN_dim),
            nn.Softmax(dim=1),
        )

        self.attention = MaskAttention(emb_dim)

        # Learnable weights for AFN-like mixing (created on CPU; moved with .to(device) later)
        self.weight = torch.nn.Parameter(torch.Tensor(self.LNN_dim, self.semantic_num_expert + self.emotion_num_expert + self.style_num_expert)).unsqueeze(0)
        stdv = 1. / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.data.uniform_(-stdv, stdv)

        # Domain memory network input dims differ per language
        if self.dataset == 'ch':
            mem_in = self.emb_dim + 47 * 5 + 48
        else:
            mem_in = self.emb_dim + 38 * 5 + 32

        self.domain_memory = MemoryNetwork(
            input_dim=mem_in, emb_dim=mem_in, domain_num=self.domain_num, memory_num=self.memory_num, device=self.device
        )

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.all_feature = {}

        self.classifier = MLP(320, mlp_dims, dropout)

    def forward(self, **kwargs):
        content = kwargs['content']               # [B, T]
        content_masks = kwargs['content_masks']   # [B, T]

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)
        category = kwargs['category']             # [B]

        # Encode tokens
        content_feature = self.bert(content, attention_mask=content_masks)[0]  # [B, T, emb]

        # Attention pooling
        gate_input_feature, _ = self.attention(content_feature, content_masks)  # [B, emb]

        # Memory attention over domains
        memory_att = self.domain_memory(torch.cat([gate_input_feature, emotion_feature, style_feature], dim=-1), category)  # [B,1,D]

        # Build general domain embedding
        domain_emb_all = self.domain_embedder(torch.arange(self.domain_num, device=content.device, dtype=torch.long))  # [D, emb]
        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)  # [B, emb]

        # Current sample domain embedding
        idxs = category.view(-1, 1).long()  # [B,1]
        domain_embedding = self.domain_embedder(idxs).squeeze(1)  # [B, emb]
        gate_input = torch.cat([domain_embedding, general_domain_embedding], dim=-1)  # [B, 2*emb]

        gate_value = self.gate(gate_input).view(content_feature.size(0), 1, self.LNN_dim)  # [B,1,L]

        # Collect experts outputs
        shared_feature = []
        for i in range(self.semantic_num_expert):
            shared_feature.append(self.content_expert[i](content_feature).unsqueeze(1))  # [B,1,320]
        for i in range(self.emotion_num_expert):
            shared_feature.append(self.emotion_expert[i](emotion_feature).unsqueeze(1))  # [B,1,320]
        for i in range(self.style_num_expert):
            shared_feature.append(self.style_expert[i](style_feature).unsqueeze(1))      # [B,1,320]
        # [B, (S+E+St), 320]
        shared_feature = torch.cat(shared_feature, dim=1)

        # AFN-like mixing
        embed_x_abs = torch.abs(shared_feature)
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        embed_x_log = torch.log1p(embed_x_afn)

        # [1, L, (S+E+St)] x [B, (S+E+St), 320] -> [B, L, 320]
        lnn_out = torch.matmul(self.weight, embed_x_log)
        lnn_exp = torch.expm1(lnn_out)
        shared_feature = lnn_exp.contiguous().view(-1, self.LNN_dim, 320)

        # [B,1,L] bmm [B,L,320] -> [B,1,320] -> [B,320]
        shared_feature = torch.bmm(gate_value, shared_feature).squeeze()

        deep_logits = self.classifier(shared_feature)  # [B,1]
        return torch.sigmoid(deep_logits.squeeze(1))   # [B]

    def save_feature(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']
        category = kwargs['category']

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)

        for index in range(all_feature.size(0)):
            domain = int(category[index].detach().cpu().numpy())
            if domain not in self.all_feature:
                self.all_feature[domain] = []
            self.all_feature[domain].append(all_feature[index].view(1, -1).detach().cpu().numpy())

    def init_memory(self):
        # Initialize KMeans centers per domain and store as memory on the right device
        for domain in self.all_feature:
            all_feature = np.concatenate(self.all_feature[domain])
            kmeans = KMeans(n_clusters=self.memory_num, init='k-means++').fit(all_feature)
            centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).to(self.device).float()  # [M, emb_all]
            self.domain_memory.domain_memory[domain] = centers

    def write(self, **kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']

        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        style_feature = kwargs['style_feature']
        category = kwargs['category']

        content_feature = self.bert(content, attention_mask=content_masks)[0]
        content_feature, _ = self.attention(content_feature, content_masks)

        all_feature = torch.cat([content_feature, emotion_feature, style_feature], dim=1)
        all_feature = norm(all_feature)
        self.domain_memory.write(all_feature, category)


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
                 semantic_num,
                 emotion_num,
                 style_num,
                 lnn_dim,
                 dataset=None,
                 early_stop=5,
                 epoches=100):
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.semantic_num = semantic_num
        self.emotion_num = emotion_num
        self.style_num = style_num
        self.lnn_dim = lnn_dim
        self.dataset = dataset

        if os.path.exists(save_param_dir):
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir
            os.makedirs(save_param_dir, exist_ok=True)

    def train(self, logger=None):
        if logger:
            logger.info('start training......')

        self.model = M3FENDModel(
            self.emb_dim, self.mlp_dims, self.dropout,
            self.semantic_num, self.emotion_num, self.style_num,
            self.lnn_dim, len(self.category_dict), dataset=self.dataset
        )
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        # Warm-up memory by scanning once
        self.model.train()
        train_data_iter = tqdm.tqdm(self.train_loader)
        for _, batch in enumerate(train_data_iter):
            batch_data = data2gpu(batch, self.use_cuda)
            _ = self.model.save_feature(**batch_data)
        self.model.init_memory()
        print('initialization finished')

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for _, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label'].float()

                optimizer.zero_grad()
                label_pred = self.model(**batch_data)
                loss = loss_fn(label_pred, label)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    self.model.write(**batch_data)

                if scheduler is not None:
                    scheduler.step()

                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            status = '[{0}] lr = {1}; average_loss = {2}'.format(epoch + 1, str(self.lr), avg_loss.item())
            if logger:
                logger.info(status)

            # Validation
            self.model.train()
            results = self.test(self.val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, 'parameter_m3fend.pkl'))
                self.best_mem = self.model.domain_memory.domain_memory
                best_metric = results['metric']
            elif mark == 'esc':
                break
            else:
                continue

        # Load best
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_m3fend.pkl')))
        self.model.domain_memory.domain_memory = self.best_mem
        results = self.test(self.test_loader)
        if logger:
            logger.info("start testing......")
            logger.info("test score: {}\n\n".format(results))
        print(results)
        return results, os.path.join(self.save_param_dir, 'parameter_m3fend.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for _, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict)
