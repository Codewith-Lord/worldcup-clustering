"""Clustering logic for the World Cup 2026 player dashboard."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/fifa_world_cup_2026_player_performance.csv"

FEATURES = [
    "goals", "assists", "shots", "key_passes", "tackles",
    "interceptions", "clearances", "successful_dribbles", "pass_accuracy",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Average each outfield player's stats across all their matches."""
    outfield = df[df["position"] != "Goalkeeper"]
    feats = outfield.groupby("player_id")[FEATURES].mean()
    info = outfield.drop_duplicates("player_id").set_index("player_id")[
        ["player_name", "team", "position"]
    ]
    return feats.join(info)


def scale(feats: pd.DataFrame):
    return StandardScaler().fit_transform(feats[FEATURES])


def silhouette_by_k(X, k_range=range(2, 9)) -> pd.Series:
    scores = {
        k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
        for k in k_range
    }
    return pd.Series(scores, name="silhouette")


def run_kmeans(feats: pd.DataFrame, X, k: int) -> pd.DataFrame:
    """Return feats plus a readable 'cluster' label and PCA coordinates."""
    out = feats.copy()
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    out["cluster_id"] = labels

    # Name each cluster by the position most of its players have.
    names = {}
    for c in range(k):
        names[c] = out.loc[out.cluster_id == c, "position"].mode()[0] + "s"
    # If a position is split across clusters, rank them by shots (A = most attacking).
    for pos in set(names.values()):
        ids = [c for c, n in names.items() if n == pos]
        if len(ids) > 1:
            ids.sort(key=lambda c: -out.loc[out.cluster_id == c, "shots"].mean())
            for i, c in enumerate(ids):
                names[c] = f"{pos} {chr(65 + i)}"
    out["cluster"] = out["cluster_id"].map(names)

    pcs = PCA(2, random_state=0).fit_transform(X)
    out["PC1"], out["PC2"] = pcs[:, 0], pcs[:, 1]
    return out


def cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    return clustered.groupby("cluster")[FEATURES].mean().round(2)


def pca_variance(X) -> list:
    """Explained-variance ratio of the 2 PCA components used for the map."""
    return list(PCA(2, random_state=0).fit(X).explained_variance_ratio_)


def cluster_centroids(clustered: pd.DataFrame) -> pd.DataFrame:
    """Centroid of each cluster in PCA space plus its size."""
    g = clustered.groupby("cluster")
    out = g[["PC1", "PC2"]].mean()
    out["players"] = g.size()
    return out
