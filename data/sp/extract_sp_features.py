"""
Paso 1/2: extrae las features de emocion y estilo de M3FEND para el corpus en espanol.

Adaptacion al espanol de las vistas originales de M3FEND (no usa ningun extractor
de FakeNewsStyle), para que la comparacion sea contra el baseline tal como se define:

- Emocion del publisher (Dual Emotion, Zhang et al. 2021), CONTENT_EMOTION_DIM dims:
    * emotion category : probabilidades de pysentimiento/robertuito-emotion (7)
    * emotion lexicon  : NRC EmoLex en espanol, 8 emociones, ratio por token (8)
    * emotion intensity: NRC Emotion Intensity en espanol, suma/tokens por emocion (8)
    * sentiment score  : probabilidades de pysentimiento/robertuito-sentiment (3)
    * auxiliares       : palabras pos/neg (NRC), emojis, '!', '?', grado, negacion,
                         pronombres 1a/2a/3a persona (spaCy), todo ratio por token (10)
- Estilo, STYLE_DIM dims: longitudes, diversidad lexica, puntuacion, mayusculas,
  digitos y distribucion POS (spaCy es_core_news_sm).

El corpus no tiene comentarios, asi que (igual que las noticias sin comentarios del
dataset en ingles) comments_emotion = 0 y emotion_gap = [content, content].

Entrada : FakeNewsStyle/data/02_corpus_clean/{train,val,test}.pkl (mismo split y
          mismo texto limpio `text_xlmr` que usa FakeNewsStyle)
Salida  : M3FEND/data/sp/features/{split}_features.pkl (dict de numpy/listas, sin
          DataFrame, para poder leerlo con el pandas del venv de M3FEND)

Uso (con el venv de FakeNewsStyle, que tiene pysentimiento y spaCy):
    ../FakeNewsStyle/venv/bin/python data/sp/extract_sp_features.py
"""
import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
import spacy
from pysentimiento import create_analyzer

BASE_DIR = Path(__file__).resolve().parent
LEX_DIR = BASE_DIR / "lexicons"
DEFAULT_CORPUS = BASE_DIR.parents[2] / "FakeNewsStyle" / "data" / "02_corpus_clean"

NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
PYSENT_EMOTIONS = ["others", "joy", "sadness", "anger", "surprise", "disgust", "fear"]
PYSENT_SENTIMENT = ["NEG", "NEU", "POS"]

DEGREE_WORDS = {
    "muy", "mucho", "mucha", "muchos", "muchas", "demasiado", "demasiada", "demasiados",
    "demasiadas", "bastante", "bastantes", "tan", "tanto", "tanta", "tantos", "tantas",
    "sumamente", "extremadamente", "totalmente", "completamente", "absolutamente",
    "increiblemente", "increíblemente", "realmente", "super", "súper", "altamente",
    "enormemente", "poco", "poca", "pocos", "pocas", "apenas", "casi", "más", "menos",
}
NEGATION_WORDS = {
    "no", "ni", "nunca", "jamás", "jamas", "nada", "nadie", "ninguno", "ninguna",
    "ningún", "ningun", "tampoco", "sin",
}

POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ",
            "PROPN", "NUM", "AUX", "INTJ"]
PUNCT_CHARS = [".", ",", ";", ":", "!", "?", "¡", "¿", '"', "(", "-", "…"]

CONTENT_EMOTION_NAMES = (
    [f"emo_cat_{e}" for e in PYSENT_EMOTIONS]
    + [f"emo_lex_{e}" for e in NRC_EMOTIONS]
    + [f"emo_int_{e}" for e in NRC_EMOTIONS]
    + [f"sent_{s}" for s in PYSENT_SENTIMENT]
    + ["aux_pos_words", "aux_neg_words", "aux_emoji", "aux_excl", "aux_quest",
       "aux_degree", "aux_negation", "aux_pron1", "aux_pron2", "aux_pron3"]
)
STYLE_NAMES = (
    ["log_n_tokens", "avg_sent_len", "avg_word_len", "type_token_ratio"]
    + [f"punct_{i}" for i in range(len(PUNCT_CHARS))]
    + ["upper_char_ratio", "allcaps_word_ratio", "digit_ratio"]
    + [f"pos_{p}" for p in POS_TAGS]
)
CONTENT_EMOTION_DIM = len(CONTENT_EMOTION_NAMES)  # 36
STYLE_DIM = len(STYLE_NAMES)  # 32

RE_WORD = re.compile(r"\w+", flags=re.UNICODE)


def load_emolex():
    """Spanish word -> vector (anger..trust, negative, positive). Traducciones repetidas: max."""
    df = pd.read_csv(LEX_DIR / "Spanish-NRC-EmoLex.txt", sep="\t", keep_default_na=False)
    cols = NRC_EMOTIONS + ["negative", "positive"]
    lex = {}
    for word, vals in zip(df["Spanish Word"].str.lower().str.strip(), df[cols].to_numpy(dtype=np.float32)):
        if not word or " " in word or word == "no translation":
            continue
        lex[word] = np.maximum(lex[word], vals) if word in lex else vals
    return lex


def load_intensity():
    """Spanish word -> vector de intensidad (8 emociones NRC). Traducciones repetidas: max."""
    df = pd.read_csv(LEX_DIR / "Spanish-NRC-Emotion-Intensity-Lexicon-v1.txt", sep="\t", keep_default_na=False)
    lex = defaultdict(lambda: np.zeros(len(NRC_EMOTIONS), dtype=np.float32))
    idx = {e: i for i, e in enumerate(NRC_EMOTIONS)}
    for word, emo, score in zip(df["Spanish Word"].str.lower().str.strip(), df["Emotion"], df["Emotion-Intensity-Score"]):
        if not word or " " in word or emo not in idx:
            continue
        lex[word][idx[emo]] = max(lex[word][idx[emo]], float(score))
    return dict(lex)


def lookup(lex, tok):
    """Busca la forma y, si no esta, el lema (el lexico esta en forma canonica)."""
    w = tok.lower_
    if w in lex:
        return lex[w]
    lemma = tok.lemma_.lower()
    return lex.get(lemma)


def probas_vec(pred, labels):
    return np.array([float(pred.probas.get(k, 0.0)) for k in labels], dtype=np.float32)


def content_emotion(doc, text, emo_pred, sent_pred, emolex, intensity):
    words = [t for t in doc if not t.is_punct and not t.is_space]
    n = max(len(words), 1)
    lex_counts = np.zeros(10, dtype=np.float32)
    int_sum = np.zeros(len(NRC_EMOTIONS), dtype=np.float32)
    for t in words:
        v = lookup(emolex, t)
        if v is not None:
            lex_counts += v
        v = lookup(intensity, t)
        if v is not None:
            int_sum += v
    lower = [t.lower_ for t in words]
    # pronombres personales/posesivos por persona gramatical (morfologia de spaCy)
    pron_person = [p for t in words if t.pos_ in ("PRON", "DET") for p in t.morph.get("Person")]
    aux = np.array([
        lex_counts[9],  # positive
        lex_counts[8],  # negative
        sum(1 for c in text if c in emoji.EMOJI_DATA),
        text.count("!") + text.count("¡"),
        text.count("?") + text.count("¿"),
        sum(w in DEGREE_WORDS for w in lower),
        sum(w in NEGATION_WORDS for w in lower),
        pron_person.count("1"),
        pron_person.count("2"),
        pron_person.count("3"),
    ], dtype=np.float32) / n
    return np.concatenate([
        probas_vec(emo_pred, PYSENT_EMOTIONS),
        lex_counts[:8] / n,
        int_sum / n,
        probas_vec(sent_pred, PYSENT_SENTIMENT),
        aux,
    ])


def style_feature(doc, text):
    words = [t for t in doc if not t.is_punct and not t.is_space]
    n = max(len(words), 1)
    n_chars = max(len(text), 1)
    sents = list(doc.sents)
    raw_words = RE_WORD.findall(text)
    letters = [c for c in text if c.isalpha()]
    pos_counts = defaultdict(int)
    for t in words:
        pos_counts[t.pos_] += 1
    feats = [
        np.log1p(len(words)),
        len(words) / max(len(sents), 1),
        sum(len(t.text) for t in words) / n,
        len({t.lower_ for t in words}) / n,
    ]
    feats += [text.count(p) / n for p in PUNCT_CHARS]
    feats += [
        sum(c.isupper() for c in letters) / max(len(letters), 1),
        sum(1 for w in raw_words if len(w) > 1 and w.isupper()) / max(len(raw_words), 1),
        sum(c.isdigit() for c in text) / n_chars,
    ]
    feats += [pos_counts[p] / n for p in POS_TAGS]
    return np.array(feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=str(DEFAULT_CORPUS))
    parser.add_argument("--out_dir", default=str(BASE_DIR / "features"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emolex, intensity = load_emolex(), load_intensity()
    print(f"EmoLex es: {len(emolex)} palabras | Intensity es: {len(intensity)} palabras")
    nlp = spacy.load("es_core_news_sm", disable=["ner"])
    nlp.max_length = 2_000_000
    emo_an = create_analyzer(task="emotion", lang="es")
    sent_an = create_analyzer(task="sentiment", lang="es")

    for split in ["train", "val", "test"]:
        df = pd.read_pickle(Path(args.corpus_dir) / f"{split}.pkl")
        texts = [str(t) for t in df["text_xlmr"].tolist()]
        emo_preds = emo_an.predict(texts)
        sent_preds = sent_an.predict(texts)
        ce, st = [], []
        for text, doc, ep, sp in zip(texts, nlp.pipe(texts, batch_size=args.batch_size), emo_preds, sent_preds):
            ce.append(content_emotion(doc, text, ep, sp, emolex, intensity))
            st.append(style_feature(doc, text))
        payload = {
            "Id": df["Id"].tolist(),
            "content": texts,
            "topic": [str(t) for t in df["Topic"].tolist()],
            "label_str": [str(l) for l in df["label"].tolist()],
            "content_emotion": np.vstack(ce),
            "style_feature": np.vstack(st),
            "content_emotion_names": CONTENT_EMOTION_NAMES,
            "style_names": STYLE_NAMES,
        }
        with open(out_dir / f"{split}_features.pkl", "wb") as f:
            pickle.dump(payload, f)
        print(f"{split}: {len(texts)} filas | content_emotion {payload['content_emotion'].shape} | style {payload['style_feature'].shape}")


if __name__ == "__main__":
    main()
