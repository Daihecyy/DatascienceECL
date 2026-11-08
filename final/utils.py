import re
from collections import Counter

import numpy as np
import pandas as pd

PATTERN_ECRAN = re.compile(r"\((.*?)\)")
PATTERN_CONF_ECRAN = re.compile(r"<(.*?)>")
PATTERN_CHAINE = re.compile(r"\$(.*?)\$")


def read_ds(ds_name: str):
    with open(f"data/{ds_name}.csv") as f:
        max_actions = max(len(str(c).split(",")) for c in f)
        f.seek(0)
        column_labels = ["util", "navigateur"] if "train" in ds_name else ["navigateur"]
        column_labels.extend(range(max_actions - len(column_labels)))
        return pd.read_csv(f, names=column_labels, dtype=str)


def to_categories(df, col="util"):
    df[[col]] = df[[col]].apply(lambda x: x.cat.codes)


def clean_action(value: str):
    for delim in ["(", "<", "$", "1"]:
        if delim in value and (low_ind := value.index(delim)):
            value = value[:low_ind]
    return value


def get_sessions(df) -> list[list[str]]:
    debut = 2 if "util" in df.columns else 1  # le test n'a pas de colonne util
    return [
        [v for v in ligne if isinstance(v, str)]
        for ligne in df.iloc[:, debut:].to_numpy()
    ]


def is_timestamp(a: str) -> bool:
    return re.fullmatch(r"t\d+", a) is not None


def normalize_string(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def get_outliers(df):
    df_length = df.iloc[:, 2:].notna().sum(axis=1)
    log_length = np.log10(df_length)
    q1, q3 = log_length.quantile([0.25, 0.75])
    iqr = q3 - q1
    borne_basse, borne_haute = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = (log_length < borne_basse) | (log_length > borne_haute)

    return outliers


def build_features(sessions):
    lignes = []
    for session in sessions:
        actions = [
            normalize_string(clean_action(a)) for a in session if not is_timestamp(a)
        ]
        temps = (int(a[1:]) for a in session if is_timestamp(a))
        ecrans = [
            normalize_string(word)
            for action in session
            for word in PATTERN_ECRAN.findall(action)
        ]
        confs = [
            normalize_string(word)
            for action in session
            for word in PATTERN_CONF_ECRAN.findall(action)
        ]
        chaines = [
            normalize_string(word)
            for action in session
            for word in PATTERN_CHAINE.findall(action)
        ]

        f = {
            "n_actions": len(actions),
            "n_actions_uniques": len(set(actions)),
            "n_ecrans": len(ecrans),
            "n_ecrans_uniques": len(set(ecrans)),
            "n_confs": len(confs),
            "n_confs_uniques": len(set(confs)),
            "n_chaines": len(chaines),
            "n_chaines_uniques": len(set(chaines)),
            "duree": max(temps, default=0),
            "actions": " ".join(actions),
            "ecrans": " ".join(ecrans),
            "confs": " ".join(confs),
            "chaines": " ".join(chaines),
        }

        lignes.append(f)

    return pd.DataFrame(lignes)
