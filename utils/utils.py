# utils.py
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import torch

class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)


def _safe_auc(y_true, y_pred):
    """Devuelve AUC redondeado o None si no se puede calcular (p.ej., una sola clase)."""
    try:
        return round(float(roc_auc_score(y_true, y_pred)), 4)
    except Exception:
        return None


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}

    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    # AUC por categoría (y_pred como score/probabilidad)
    for c, res in res_by_category.items():
        metrics_by_category[c] = {
            'auc': _safe_auc(res['y_true'], res['y_pred'])
        }

    # Métricas globales
    try:
        metrics_by_category['auc'] = float(roc_auc_score(y_true, y_pred))
    except Exception:
        metrics_by_category['auc'] = None

    # Binariza predicciones una sola vez
    y_pred_bin = np.around(np.array(y_pred)).astype(int)

    metrics_by_category['metric']   = round(float(f1_score(y_true, y_pred_bin, average='macro')), 4)
    metrics_by_category['recall']   = round(float(recall_score(y_true, y_pred_bin, average='macro')), 4)
    metrics_by_category['precision']= round(float(precision_score(y_true, y_pred_bin, average='macro')), 4)
    metrics_by_category['acc']      = round(float(accuracy_score(y_true, y_pred_bin)), 4)

    # Métricas por categoría con redondeo correcto (sin .tolist())
    for c, res in res_by_category.items():
        y_pred_c_bin = np.around(np.array(res['y_pred'])).astype(int)
        metrics_by_category[c] = {
            'precision': round(float(precision_score(res['y_true'], y_pred_c_bin, average='macro')), 4),
            'recall':    round(float(recall_score(res['y_true'], y_pred_c_bin, average='macro')), 4),
            'fscore':    round(float(f1_score(res['y_true'], y_pred_c_bin, average='macro')), 4),
            'auc':       metrics_by_category[c]['auc'],
            'acc':       round(float(accuracy_score(res['y_true'], y_pred_c_bin)), 4),
        }
    return metrics_by_category


def data2gpu(batch, use_cuda):
    """
    Cambio mínimo: reemplaza .cuda() por .to(device) sin tocar la firma.
    """
    device = torch.device('cuda' if use_cuda else 'cpu')
    return {
        'content':           batch[0].to(device),
        'content_masks':     batch[1].to(device),
        'comments':          batch[2].to(device),
        'comments_masks':    batch[3].to(device),
        'content_emotion':   batch[4].to(device),
        'comments_emotion':  batch[5].to(device),
        'emotion_gap':       batch[6].to(device),
        'style_feature':     batch[7].to(device),
        'label':             batch[8].to(device),
        'category':          batch[9].to(device),
    }


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
