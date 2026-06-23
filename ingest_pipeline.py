#!/usr/bin/env python3
"""
================================================================
  UMTS KPI — Pipeline d'ingestion vers Elasticsearch
  Rôle : lire data.csv → feature engineering → prédictions ML
         → indexation directe dans Elasticsearch
         → export JSONL pour Filebeat (archivage/replay)

  Dépendances :
    pandas==2.2.2  numpy==1.26.4  scikit-learn==1.4.2
    elasticsearch==8.13.2  joblib==1.4.2

  Variables d'environnement (avec valeurs par défaut) :
    ELASTICSEARCH_HOST  http://elasticsearch:9200
    DATA_PATH           /notebooks/data/data.csv
    MODEL_PATH          /models
    PREP_PATH           /notebooks/data/data_prepared
================================================================
"""

import os, sys, json, time, logging, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("umts-ingest")

# ── Config ─────────────────────────────────────────────────────
ES_HOST        = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
DATA_PATH      = Path(os.getenv("DATA_PATH",  "/notebooks/data/data.csv"))
MODEL_DIR      = Path(os.getenv("MODEL_PATH", "/models"))
PREP_DIR       = Path(os.getenv("PREP_PATH",  "/notebooks/data/data_prepared"))
OUT_DIR        = Path("/data/incoming")
BATCH_SIZE     = 500
RSCP_THRESHOLD = -85.0   # dBm — seuil cible ML

# ══════════════════════════════════════════════════════════════
#  STEP 1 — Elasticsearch : connexion + setup
# ══════════════════════════════════════════════════════════════

def wait_for_es(es: Elasticsearch, timeout: int = 180) -> None:
    """Attend qu'Elasticsearch réponde. Quitte si timeout dépassé."""
    log.info(f"Connexion Elasticsearch ({ES_HOST}) ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            info = es.info()
            log.info(f"✅ ES {info['version']['number']} disponible")
            return
        except Exception:
            time.sleep(5)
    log.error("❌ Elasticsearch indisponible après %ds", timeout)
    sys.exit(1)


def setup_index_template(es: Elasticsearch) -> None:
    """
    Crée un index template avec mapping explicite.
    Idempotent — peut être relancé sans risque.
    """
    tpl = {
        "index_patterns": ["umts-kpi-*"],
        "template": {
            "settings": {
                "number_of_shards":   1,
                "number_of_replicas": 0,        # mono-node
                "refresh_interval":   "10s",
                "index.codec":        "best_compression",
            },
            "mappings": {
                "dynamic": True,
                "_source": {"enabled": True},
                "properties": {
                    "@timestamp":           {"type": "date"},
                    "band":                 {"type": "keyword"},
                    "channel_1":            {"type": "integer"},
                    "sc_active_1":          {"type": "integer"},
                    "rscp_active_1":        {"type": "float"},
                    "rscp_active_2":        {"type": "float"},
                    "rscp_active_mean":     {"type": "float"},
                    "rscp_active_max":      {"type": "float"},
                    "rscp_active_min":      {"type": "float"},
                    "rscp_active_std":      {"type": "float"},
                    "n_cells_active":       {"type": "integer"},
                    "ecno_active_1":        {"type": "float"},
                    "ecno_active_mean":     {"type": "float"},
                    "ecno_active_max":      {"type": "float"},
                    "ecno_active_min":      {"type": "float"},
                    "rscp_detected_1":      {"type": "float"},
                    "rscp_detected_mean":   {"type": "float"},
                    "rscp_detected_max":    {"type": "float"},
                    "n_cells_detected":     {"type": "integer"},
                    "rscp_class":           {"type": "keyword"},
                    "quality_score":        {"type": "float"},
                    "is_anomaly":           {"type": "boolean"},
                    "anomaly_flag":         {"type": "integer"},
                    "pred_class":           {"type": "integer"},
                    "pred_proba":           {"type": "float"},
                    "pred_label":           {"type": "keyword"},
                    "pred_method":          {"type": "keyword"},
                    "data_source":          {"type": "keyword"},
                    "project":              {"type": "keyword"},
                    "ingestion_ts":         {"type": "date"},
                }
            }
        },
        "priority": 100,
        "_meta": {"description": "UMTS KPI Analytics — PFE"}
    }
    try:
        es.indices.put_index_template(name="umts-kpi-template", body=tpl)
        log.info("✅ Index template créé : umts-kpi-*")
    except Exception as exc:
        log.warning("Template (skip si déjà existant) : %s", exc)


# ══════════════════════════════════════════════════════════════
#  STEP 2 — Chargement CSV
# ══════════════════════════════════════════════════════════════

def load_csv(path: Path) -> pd.DataFrame:
    log.info("Chargement CSV : %s", path)
    if not path.exists():
        log.error("Fichier introuvable : %s", path)
        sys.exit(1)

    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        na_values=["", "n/a", "N/A", "NA", "nan", "NaN", "null", "n\\a"],
        keep_default_na=True,
        encoding="utf-8",
        encoding_errors="replace",
    )
    df.columns = df.columns.str.strip()
    # Supprimer les lignes entièrement vides
    df = df.dropna(how="all").reset_index(drop=True)
    log.info("  → %d lignes × %d colonnes", len(df), len(df.columns))
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 3 — Feature Engineering
# ══════════════════════════════════════════════════════════════

def _parse_mv_float(series: pd.Series) -> list:
    """Chaque cellule → liste de float (multi-valeurs séparées par virgule)."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append([])
            continue
        items = []
        for p in str(v).split(","):
            try:
                items.append(float(p.strip()))
            except ValueError:
                pass
        out.append(items)
    return out

def _parse_mv_int(series: pd.Series) -> list:
    out = []
    for v in series:
        if pd.isna(v):
            out.append([])
            continue
        items = []
        for p in str(v).split(","):
            try:
                items.append(int(float(p.strip())))
            except ValueError:
                pass
        out.append(items)
    return out


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    RENAME = {
        "Time":                       "time_raw",
        "Band (active)":              "band",
        "Channel number (active)":    "channel_raw",
        "Scrambling code (active)":   "sc_active_raw",
        "RSCP (active)":              "rscp_active_raw",
        "Ec/N0 (active)":             "ecno_active_raw",
        "RSCP (detected)":            "rscp_detected_raw",
        "Scrambling code (detected)": "sc_detected_raw",
    }
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    out = pd.DataFrame(index=df.index)

    # ── Identifiants ─────────────────────────────────────────
    if "time_raw" in df.columns:
        out["time_raw"] = df["time_raw"]
    if "band" in df.columns:
        out["band"] = df["band"].str.strip().str.upper().fillna("UNKNOWN")

    # ── RSCP actif ───────────────────────────────────────────
    if "rscp_active_raw" in df.columns:
        vals = _parse_mv_float(df["rscp_active_raw"])
        out["rscp_active_1"]    = [v[0] if v else np.nan for v in vals]
        out["rscp_active_2"]    = [v[1] if len(v) > 1 else np.nan for v in vals]
        out["rscp_active_mean"] = [round(np.mean(v), 2) if v else np.nan for v in vals]
        out["rscp_active_max"]  = [round(max(v), 2)  if v else np.nan for v in vals]
        out["rscp_active_min"]  = [round(min(v), 2)  if v else np.nan for v in vals]
        out["rscp_active_std"]  = [round(float(np.std(v)), 2) if len(v) > 1 else 0.0 for v in vals]
        out["n_cells_active"]   = [len(v) for v in vals]

    # ── Ec/N0 actif ──────────────────────────────────────────
    if "ecno_active_raw" in df.columns:
        vals = _parse_mv_float(df["ecno_active_raw"])
        out["ecno_active_1"]    = [v[0] if v else np.nan for v in vals]
        out["ecno_active_mean"] = [round(np.mean(v), 2) if v else np.nan for v in vals]
        out["ecno_active_max"]  = [round(max(v), 2)  if v else np.nan for v in vals]
        out["ecno_active_min"]  = [round(min(v), 2)  if v else np.nan for v in vals]

    # ── RSCP détecté ─────────────────────────────────────────
    if "rscp_detected_raw" in df.columns:
        vals = _parse_mv_float(df["rscp_detected_raw"])
        out["rscp_detected_1"]    = [v[0] if v else np.nan for v in vals]
        out["rscp_detected_mean"] = [round(np.mean(v), 2) if v else np.nan for v in vals]
        out["rscp_detected_max"]  = [round(max(v), 2)  if v else np.nan for v in vals]
        out["n_cells_detected"]   = [len(v) for v in vals]

    # ── Channel & Scrambling code ─────────────────────────────
    if "channel_raw" in df.columns:
        vals = _parse_mv_int(df["channel_raw"])
        out["channel_1"] = [v[0] if v else np.nan for v in vals]
    if "sc_active_raw" in df.columns:
        vals = _parse_mv_int(df["sc_active_raw"])
        out["sc_active_1"] = [v[0] if v else np.nan for v in vals]

    log.info("  Feature engineering → %d colonnes", out.shape[1])
    return out


# ══════════════════════════════════════════════════════════════
#  STEP 4 — Timestamp
# ══════════════════════════════════════════════════════════════

_TS_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]

def _parse_one_ts(raw) -> str:
    if pd.isna(raw) or str(raw).strip() == "":
        return datetime.now(timezone.utc).isoformat()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(str(raw).strip(), fmt).replace(
                tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()

def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "time_raw" in df.columns:
        df["@timestamp"] = df["time_raw"].apply(_parse_one_ts)
    else:
        df["@timestamp"] = datetime.now(timezone.utc).isoformat()
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 5 — Prédictions ML
# ══════════════════════════════════════════════════════════════

def load_model_and_prep(model_dir: Path, prep_dir: Path):
    """
    Charge le meilleur modèle (.pkl) + preprocessor + metadata.
    Retourne (model, preprocessor, metadata) ou (None, None, {}).
    """
    try:
        import joblib
    except ImportError:
        log.warning("joblib non installé — prédictions ML désactivées")
        return None, None, {}

    # Chercher best_model_*.pkl en priorité
    candidates = sorted(model_dir.glob("best_model_*.pkl"))
    if not candidates:
        candidates = sorted(model_dir.glob("model_*.pkl"))
    if not candidates:
        candidates = sorted(model_dir.glob("*.pkl"))
        # Exclure le preprocessor
        candidates = [p for p in candidates if "preprocessor" not in p.name]

    if not candidates:
        log.warning("Aucun modèle .pkl trouvé dans %s", model_dir)
        return None, None, {}

    model_path = candidates[0]
    log.info("Modele detecte : %s", model_path.name)
    try:
        model = joblib.load(model_path)
        log.info("Modele charge avec succes")
    except Exception as exc:
        log.warning("Chargement modele impossible (%s) -> fallback rule_based active", exc)
        return None, None, {}

    # Preprocessor
    prep = None
    for pp in [
        prep_dir / "preprocessor.pkl",
        model_dir / "preprocessor.pkl",
        model_dir.parent / "data" / "data_prepared" / "preprocessor.pkl",
    ]:
        if pp.exists():
            try:
                prep = joblib.load(pp)
                log.info("Preprocessor charge : %s", pp)
                break
            except Exception as exc:
                log.warning("Preprocessor incompatible (%s): %s", pp, exc)
    if prep is None:
        log.warning("Aucun preprocessor trouvé — features brutes utilisées")

    # Metadata (liste des features d'entraînement)
    meta = {}
    for mp in [
        prep_dir / "metadata.json",
        model_dir / "metadata.json",
    ]:
        if mp.exists():
            with open(mp) as f:
                meta = json.load(f)
            log.info("Metadata chargée : %s", mp)
            break

    return model, prep, meta


def apply_predictions(df: pd.DataFrame, model, prep, meta: dict) -> pd.DataFrame:
    """Applique le modèle ML sur df et ajoute les colonnes pred_*."""
    if model is None:
        return _rule_based_predictions(df)

    try:
        num_feats = meta.get("num_features", [])
        cat_feats = meta.get("cat_features", [])
        all_feats = num_feats + cat_feats

        # Garder seulement les features disponibles dans df
        available = [f for f in all_feats if f in df.columns]
        if not available:
            # Fallback : toutes les numériques
            available = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c not in ("@timestamp", "anomaly_flag")]

        X = df[available].copy()

        # Conversion numérique
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        # Imputation simple (médiane) — cohérent avec notebook 03
        for c in X.columns:
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median())

        if prep is not None:
            X_prep = prep.transform(X)
        else:
            X_prep = X.values

        preds  = model.predict(X_prep)
        probas = (model.predict_proba(X_prep)[:, 1]
                  if hasattr(model, "predict_proba") else preds.astype(float))

        df["pred_class"]  = preds.astype(int)
        df["pred_proba"]  = probas.round(4)
        df["pred_label"]  = pd.Series(preds).map({1: "good", 0: "poor"}).values
        df["pred_method"] = "ml_model"

        n_good = int((preds == 1).sum())
        n_poor = int((preds == 0).sum())
        log.info("  ML → bonne qualité: %d | mauvaise: %d", n_good, n_poor)

    except Exception as exc:
        log.warning("Erreur ML (%s) → fallback règle métier", exc)
        df = _rule_based_predictions(df)

    return df


def _rule_based_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rscp = pd.to_numeric(df.get("rscp_active_1", pd.Series([np.nan] * len(df))), errors="coerce")
    df["pred_class"]  = (rscp >= RSCP_THRESHOLD).astype(int)
    df["pred_proba"]  = np.where(rscp >= RSCP_THRESHOLD, 0.75, 0.25)
    df["pred_label"]  = df["pred_class"].map({1: "good", 0: "poor"})
    df["pred_method"] = "rule_based"
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 6 — Enrichissement final
# ══════════════════════════════════════════════════════════════

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute qualité signal, anomalie, métadonnées."""
    rscp = pd.to_numeric(df.get("rscp_active_1", pd.Series([np.nan] * len(df))), errors="coerce")
    ecno = pd.to_numeric(df.get("ecno_active_1", pd.Series([np.nan] * len(df))), errors="coerce")

    def _cls(v):
        if pd.isna(v): return "unknown"
        if v >= -75:   return "excellent"
        if v >= -85:   return "good"
        if v >= -95:   return "fair"
        if v >= -105:  return "poor"
        return "very_poor"

    df["rscp_class"]    = rscp.apply(_cls)
    df["quality_score"] = ((rscp + 120) * 2).clip(0, 100).round(1)
    df["is_anomaly"]    = (rscp < -95) | (ecno < -12)
    df["anomaly_flag"]  = df["is_anomaly"].astype(int)
    df["ingestion_ts"]  = datetime.now(timezone.utc).isoformat()
    df["data_source"]   = "umts_kpi"
    df["project"]       = "pfe_umts_monitoring"
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 7 — Indexation Elasticsearch
# ══════════════════════════════════════════════════════════════

def _row_to_doc(row: pd.Series) -> dict:
    """Convertit une ligne en document ES — ignore les NaN."""
    doc = {}
    for k, v in row.items():
        # Ignorer NaN
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        # Cast types numpy → Python natif
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        doc[k] = v
    return doc


def index_dataframe(es: Elasticsearch, df: pd.DataFrame, index: str) -> tuple:
    log.info("Indexation → %s (%d docs, batch=%d)", index, len(df), BATCH_SIZE)

    actions = (
        {"_index": index, "_source": _row_to_doc(row)}
        for _, row in df.iterrows()
    )

    ok, errors = 0, 0
    for i in range(0, len(df), BATCH_SIZE):
        batch = [
            {"_index": index, "_source": _row_to_doc(row)}
            for _, row in df.iloc[i:i + BATCH_SIZE].iterrows()
        ]
        try:
            n_ok, n_err = helpers.bulk(es, batch, raise_on_error=False, request_timeout=60)
            ok     += n_ok
            errors += len(n_err) if isinstance(n_err, list) else n_err
            pct = min(100, (i + len(batch)) / len(df) * 100)
            log.info("  batch %d/%d → %d docs (%d%%)",
                     i // BATCH_SIZE + 1, -(-len(df) // BATCH_SIZE), n_ok, int(pct))
        except Exception as exc:
            log.error("  Erreur batch %d : %s", i // BATCH_SIZE + 1, exc)
            errors += len(batch)

    log.info("✅ Indexation : %d OK | %d erreurs", ok, errors)
    return ok, errors


# ══════════════════════════════════════════════════════════════
#  STEP 8 — Export JSONL (pour Filebeat / archivage)
# ══════════════════════════════════════════════════════════════

def export_jsonl(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"umts_kpi_{ts}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            doc = _row_to_doc(row)
            f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")

    log.info("✅ JSONL exporté : %s (%d lignes)", path, len(df))
    return path


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("  UMTS KPI Ingestion Pipeline")
    log.info("  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 55)

    # 1 — Connexion ES
    es = Elasticsearch(ES_HOST, request_timeout=30)
    wait_for_es(es)
    setup_index_template(es)

    # 2 — Chargement CSV
    df_raw = load_csv(DATA_PATH)

    # 3 — Feature engineering
    df = feature_engineering(df_raw)

    # 4 — Timestamp
    df = add_timestamp(df)

    # 5 — Modèle ML
    model, prep, meta = load_model_and_prep(MODEL_DIR, PREP_DIR)
    df = apply_predictions(df, model, prep, meta)

    # 6 — Enrichissement
    df = enrich(df)

    # Supprimer time_raw (déjà dans @timestamp)
    df = df.drop(columns=["time_raw"], errors="ignore")

    log.info("Dataset final : %d lignes × %d colonnes", *df.shape)

    # 7 — Export JSONL
    jsonl_path = export_jsonl(df)

    # 8 — Indexation ES
    today = datetime.now().strftime("%Y.%m.%d")
    index = f"umts-kpi-{today}"
    n_ok, n_err = index_dataframe(es, df, index)

    # 9 — Résumé
    log.info("=" * 55)
    log.info("  TERMINÉ")
    log.info("  Index ES  : %s", index)
    log.info("  Succès    : %d | Erreurs : %d", n_ok, n_err)
    log.info("  JSONL     : %s", jsonl_path)
    log.info("  Kibana    : http://localhost:5601")
    log.info("=" * 55)

    if n_ok == 0:
        log.error("❌ Aucun document indexé — vérifier Elasticsearch et les données")
        sys.exit(1)


if __name__ == "__main__":
    main()