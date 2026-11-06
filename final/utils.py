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


def filter_action(value: str):
    for delim in ["(", "<", "$", "1"]:
        if delim in value and (low_ind := value.index(delim)):
            value = value[:low_ind]
    return value


def get_outliers(df):
    df_length = df.iloc[:, 2:].notna().sum(axis=1)
    log_length = np.log10(df_length)
    q1, q3 = log_length.quantile([0.25, 0.75])
    iqr = q3 - q1
    borne_basse, borne_haute = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = (log_length < borne_basse) | (log_length > borne_haute)

    return outliers


def get_sessions(df):
    debut = 2 if "util" in df.columns else 1  # le test n'a pas de colonne util
    return [
        [v for v in ligne if isinstance(v, str)]
        for ligne in df.iloc[:, debut:].to_numpy()
    ]


def is_timestamp(a):
    return re.fullmatch(r"t\d+", a) is not None


def build_features(sessions, filtered_uniques):
    lignes = []
    for s in sessions:
        actions = [a for a in s if not is_timestamp(a)]
        temps = [int(a[1:]) for a in s if is_timestamp(a)]
        n_actions = len(actions)

        f = {
            "n_actions": len(actions),
            "n_actions_uniques": len(set(actions)),
            "duree": max(temps, default=0),
        }

        compte = Counter(filter_action(a) for a in actions)
        for type_action in filtered_uniques:
            f["freq_" + type_action] = compte.get(type_action, 0) / n_actions

        lignes.append(f)

    return pd.DataFrame(lignes)


def most_frequent(valeurs):
    return Counter(valeurs).most_common(1)[0][0] if valeurs else "aucun"


def regex_features(sessions):
    lignes = []
    for s in sessions:
        ecrans = [m for a in s for m in PATTERN_ECRAN.findall(a)]
        confs = [m for a in s for m in PATTERN_CONF_ECRAN.findall(a)]
        chaines = [m for a in s for m in PATTERN_CHAINE.findall(a)]

        lignes.append(
            {
                "n_ecrans_total": len(ecrans),
                "n_confs_total": len(confs),
                "n_chaines_total": len(chaines),
                "ecran_top": most_frequent(ecrans),
                "conf_top": most_frequent(confs),
                "chaine_top": most_frequent(chaines),
            }
        )

    return pd.DataFrame(lignes)
