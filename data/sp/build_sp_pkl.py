"""
Paso 2/2: arma los train/val/test.pkl de M3FEND a partir de las features del paso 1.

Formato identico al de data/en y data/ch (DataFrame con 8 columnas):
    content, comments, category, label, content_emotion, comments_emotion,
    emotion_gap, style_feature
El corpus no tiene comentarios: comments = "", comments_emotion = 0 (2*D) y
emotion_gap = [content - mean(comments), content - max(comments)] = [content, content].

Dominios: se usa `Topic`, agrupando en "Other" los que tienen muy pocas noticias
(el Domain Memory Bank de M3FEND corre KMeans con 10 memorias por dominio sobre train,
asi que un dominio necesita >= 10 noticias en train).

Salidas:
    data/sp/{train,val,test}.pkl        -> features de emocion/estilo reales
    data/sp_zeros/{train,val,test}.pkl  -> mismas filas con emocion/estilo = 0 (ablacion)

Uso (con el venv de M3FEND):
    .venv/bin/python data/sp/build_sp_pkl.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = BASE_DIR / "features"
ZEROS_DIR = BASE_DIR.parent / "sp_zeros"

OTHER_TOPICS = {"Environment", "International", "Education"}


def build(split):
    with open(FEATURES_DIR / f"{split}_features.pkl", "rb") as f:
        d = pickle.load(f)
    ce = d["content_emotion"].astype(np.float32)
    n = ce.shape[0]
    comments_emotion = np.zeros((n, 2 * ce.shape[1]), dtype=np.float32)
    emotion_gap = np.concatenate([ce, ce], axis=1)
    label_map = {"Fake": 1, "True": 0}
    df = pd.DataFrame({
        "content": d["content"],
        "comments": [""] * n,
        "category": ["Other" if t in OTHER_TOPICS else t for t in d["topic"]],
        "label": [label_map[l] for l in d["label_str"]],
        "content_emotion": list(ce),
        "comments_emotion": list(comments_emotion),
        "emotion_gap": list(emotion_gap),
        "style_feature": list(d["style_feature"].astype(np.float32)),
    })
    df = df.astype({"content": object, "comments": object, "category": object, "label": object})
    return df


def main():
    ZEROS_DIR.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        df = build(split)
        df.to_pickle(BASE_DIR / f"{split}.pkl")

        zeros = df.copy()
        for col in ["content_emotion", "comments_emotion", "emotion_gap", "style_feature"]:
            zeros[col] = [np.zeros_like(v) for v in df[col]]
        zeros.to_pickle(ZEROS_DIR / f"{split}.pkl")

        print(f"{split}: {df.shape} | label {df['label'].value_counts().to_dict()}")
        print(f"  category {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
