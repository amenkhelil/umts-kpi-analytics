# UMTS KPI Analytics

Projet académique réalisé dans le cadre du module "Ingénierie des réseaux mobiles" à ENET'COM, portant sur l'analyse des KPIs radio UMTS par le Big Data et le Machine Learning.

L'objectif est de traiter des mesures de drive test collectées sur le réseau UMTS FDD 2100 MHz, d'évaluer la qualité du signal (RSCP, Ec/N0), de classifier les enregistrements par classe de qualité via des modèles ML, et de visualiser les résultats dans un dashboard Kibana temps réel.

## Dataset

Données de drive test issues de mesures terrain sur réseau UMTS FDD bande 2100 MHz, disponibles sur [Kaggle](https://www.kaggle.com/datasets/mariamdhieb/telecom-dataset).

Colonnes principales :

| Colonne | Description |
|---|---|
| `Time` | Horodatage de la mesure |
| `Band (active)` | Bande radio active (UMTS FDD 2100) |
| `Channel number (active)` | Numéro de canal UARFCN |
| `Scrambling code (active)` | Code de brouillage de la cellule active |
| `RSCP (active)` | Received Signal Code Power en dBm |
| `Ec/N0 (active)` | Rapport signal/bruit en dB |
| `RSCP (detected)` | RSCP de la cellule détectée (voisine) |
| `Scrambling code (detected)` | Code de brouillage de la cellule détectée |

## Ce que fait le projet

- Stack Docker Compose complète : Elasticsearch, Kibana, Logstash, Filebeat, ingestion Python
- Pipeline d'ingestion Python : lecture CSV, feature engineering sur RSCP et Ec/N0, prédiction ML, export JSONL, indexation Elasticsearch
- Pipeline Logstash pour le replay Filebeat de fichiers JSONL ou CSV bruts
- Notebooks Jupyter pour l'exploration, le preprocessing et l'entraînement des modèles
- Index Elasticsearch quotidiens avec le pattern `umts-kpi-YYYY.MM.dd`
- Modèle de données Kibana-ready : classes de qualité signal, flags d'anomalie, labels de prédiction, scores

## Architecture

```
notebooks/data/data.csv
        |
        v
Python ingestion container
        |-- feature engineering (RSCP, Ec/N0)
        |-- ML ou rule-based prediction
        |-- export JSONL vers data/incoming
        v
Elasticsearch index: umts-kpi-YYYY.MM.dd
        |
        v
Kibana dashboards / Discover

data/incoming/*.jsonl ou *.csv
        |
        v
Filebeat -> Logstash -> Elasticsearch
```

## Stack technique

- Python 3.11, pandas, NumPy, scikit-learn, joblib, Elasticsearch Python client
- Jupyter Notebook, matplotlib, seaborn, XGBoost
- Docker Compose
- Elastic Stack 8.13.4 : Elasticsearch, Kibana, Logstash, Filebeat

## Prérequis

- Docker Engine avec Docker Compose v2
- Au minimum 4 Go de RAM alloués à Docker
- Python 3.11+ pour exécuter les notebooks ou le script d'ingestion hors Docker
- Git

## Installation

```bash
git clone https://github.com/amenkhelil/umts-kpi-analytics.git
cd umts-kpi-analytics
```

Pour le travail Python local :

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Configuration

Variables d'environnement utilisées par le conteneur d'ingestion :

| Variable             | Description                                      | Défaut                        |
|----------------------|--------------------------------------------------|-------------------------------|
| `ELASTICSEARCH_HOST` | URL Elasticsearch depuis le conteneur            | `http://elasticsearch:9200`   |
| `DATA_PATH`          | Chemin du CSV dans le conteneur                  | `/project-data/data.csv`      |
| `MODEL_PATH`         | Répertoire des modèles ML                        | `/models`                     |
| `PREP_PATH`          | Répertoire preprocesseur/métadonnées             | `/project-data/data_prepared` |
| `PYTHONUNBUFFERED`   | Logs Python en temps réel                        | `1`                           |

## Lancer le projet

```bash
# Démarrer la stack complète
docker compose up -d

# Suivre les logs d'ingestion
docker compose logs -f ingestion

# Ouvrir Kibana
http://localhost:5601
```

Créer une data view Kibana avec le pattern d'index : `umts-kpi-*`

```bash
# Arrêter la stack
docker compose down

# Supprimer les volumes Elastic (repart de zéro)
docker compose down -v
```

## Commandes utiles

```bash
# Lancer uniquement l'ingestion Python
docker compose up ingestion

# Ingestion hors Docker
export ELASTICSEARCH_HOST=http://localhost:9200
export DATA_PATH=notebooks/data/data.csv
export MODEL_PATH=notebooks/models
export PREP_PATH=notebooks/data/data_prepared
python ingest_pipeline.py

# Vérifier les documents indexés
curl "http://localhost:9200/umts-kpi-*/_search?size=5&pretty"

# Vérifier la santé du cluster
curl "http://localhost:9200/_cluster/health?pretty"
```

## Endpoints locaux

| Service       | URL                        |
|---------------|----------------------------|
| Elasticsearch | http://localhost:9200      |
| Kibana        | http://localhost:5601      |
| Logstash API  | http://localhost:9600      |
| Filebeat      | envoie vers logstash:5044  |

## Structure du projet

```
.
├── docker-compose.yml        # Stack Elastic locale + service d'ingestion
├── elasticsearch.yml         # Config Elasticsearch
├── kibana.yml                # Config Kibana
├── logstash.yml              # Config runtime Logstash
├── umts_kpi.conf             # Pipeline Logstash (parsing + indexation)
├── filebeat.yml              # Config Filebeat filestream
├── ingest_pipeline.py        # Feature engineering, prédiction, indexation
├── requirements.txt          # Dépendances Python
├── data/incoming/            # Fichiers JSONL exportés pour replay Filebeat
├── notebooks/                # Notebooks d'exploration, preprocessing, entraînement
├── notebooks/data/           # Données brutes et préparées
└── notebooks/models/         # Modèles entraînés
```

## Sécurité

La sécurité Elasticsearch et le TLS sont désactivés pour le développement local. Ne pas exposer cette stack sur un réseau partagé sans activer l'authentification Elastic et le TLS.

## Troubleshooting

**Elasticsearch unhealthy**
- Augmenter la RAM Docker à au moins 4 Go
- `docker compose logs elasticsearch`
- `docker compose down -v` si l'état du cluster est corrompu

**Kibana ne démarre pas**
- Attendre qu'Elasticsearch soit healthy
- `docker compose logs kibana`
- Vérifier que le port 5601 est libre

**Pas de données dans Kibana**
- `docker compose logs ingestion`
- Vérifier que `notebooks/data/data.csv` existe
- Créer la data view `umts-kpi-*` dans Kibana
- `curl "http://localhost:9200/umts-kpi-*/_count?pretty"`

**Filebeat/Logstash ne fonctionne pas**
- Vérifier que des fichiers existent dans `data/incoming`
- `docker compose logs filebeat logstash`