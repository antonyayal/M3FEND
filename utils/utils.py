# M3FEND/utils/utils.py

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def data2gpu(batch, use_cuda: bool):
    """
    Convierte el batch a CPU o CUDA según disponibilidad/flag.

    Espera un batch con el siguiente orden (coincide con tu dataloader):
        (content, content_masks, content_emotion, comments_emotion,
         emotion_gap, style_feature, label, category)

    Devuelve un diccionario con tensores en el dispositivo adecuado.
    """
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    (content,
     content_masks,
     content_emotion,
     comments_emotion,
     emotion_gap,
     style_feature,
     label,
     category) = batch

    def to_dev(x):
        # Solo movemos tensores; otros tipos pasan tal cual
        return x.to(device, non_blocking=(use_cuda and torch.cuda.is_available())) if torch.is_tensor(x) else x

    return {
        "content": to_dev(content),
        "content_masks": to_dev(content_masks),
        "content_emotion": to_dev(content_emotion),
        "comments_emotion": to_dev(comments_emotion),
        "emotion_gap": to_dev(emotion_gap),
        "style_feature": to_dev(style_feature),
        "label": to_dev(label),
        "category": to_dev(category),
    }


class Averager:
    """Promedia valores incrementalmente (por ejemplo, pérdidas por batch)."""

    def __init__(self):
        self.n = 0
        self.v = 0.0

    def add(self, x: float, n: int = 1):
        self.v += float(x) * n
        self.n += n

    def item(self) -> float:
        if self.n == 0:
            return 0.0
        return self.v / self.n

    def reset(self):
        self.n = 0
        self.v = 0.0


def _safe_auc(y_true: List[int], y_prob: List[float]) -> Optional[float]:
    """Calcula AUC si hay al menos dos clases; si no, devuelve None."""
    try:
        if len(set(y_true)) < 2:
            return None
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return None


def metrics(
    labels: List[int],
    preds_prob: List[float],
    categories: List[int],
    category_dict: Dict[int, str],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Calcula métricas globales (y por categoría opcionalmente) para binario.

    Parámetros:
        labels        : lista de 0/1 reales
        preds_prob    : lista de probabilidades (0..1)
        categories    : lista de ids de dominio (mismo largo)
        category_dict : dict {id:int -> nombre:str}
        threshold     : umbral para binarizar predicciones

    Retorna:
        {
            "metric": f1_macro_o_weighted (se usa para early stopping, aquí F1 binario),
            "f1": f1_binario,
            "precision": precision_binaria,
            "recall": recall_binaria,
            "acc": accuracy,
            "auc": auc_binaria_o_None,
            "per_category": {
                "<nombre_cat>": {"f1":..., "acc":..., "precision":..., "recall":..., "auc":...}, ...
            }
        }
    """
    y_true = np.asarray(labels).astype(int)
    y_prob = np.asarray(preds_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Métricas globales
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = _safe_auc(y_true.tolist(), y_prob.tolist())

    # Métrica principal para early stopping
    main_metric = f1

    # Métricas por categoría (opcional)
    per_category = {}
    if categories is not None and category_dict is not None:
        cats = np.asarray(categories).astype(int)
        for cid, cname in category_dict.items():
            idx = np.where(cats == cid)[0]
            if idx.size == 0:
                continue
            yt = y_true[idx]
            yp = y_pred[idx]
            ypp = y_prob[idx]
            per_category[cname] = {
                "f1": f1_score(yt, yp, zero_division=0),
                "acc": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "auc": _safe_auc(yt.tolist(), ypp.tolist()),
                "support": int(idx.size),
            }

    return {
        "metric": float(main_metric),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "acc": float(acc),
        "auc": (None if auc is None else float(auc)),
        "per_category": per_category,
    }


class Recorder:
    """
    Early stopping + guardado del mejor resultado.
    Usa results['metric'] como referencia. Devuelve:
        - 'save' cuando mejora
        - 'esc'  cuando supera paciencia
        - None   en otro caso
    """

    def __init__(self, early_stop: int = 5, maximize: bool = True):
        self.early_stop = early_stop
        self.maximize = maximize
        self.best = None
        self.count = 0

    def add(self, results: Dict[str, object]) -> Optional[str]:
        cur = results.get("metric", None)
        if cur is None:
            # Si no hay métrica, no hacemos nada especial.
            return None

        if self.best is None:
            self.best = cur
            self.count = 0
            return "save"

        improved = (cur > self.best) if self.maximize else (cur < self.best)
        if improved:
            self.best = cur
            self.count = 0
            return "save"
        else:
            self.count += 1
            if self.count >= self.early_stop:
                return "esc"
            return None
