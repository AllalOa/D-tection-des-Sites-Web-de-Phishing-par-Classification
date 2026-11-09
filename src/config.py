"""
config.py - Configuration centralisée du projet PROJ-IASD01.

Tous les chemins sont relatifs à la racine du projet PROJ-IASD01,
calculés via pathlib. Aucun chemin absolu en dur.
random_state=42 partout pour la reproductibilité.

Source des colonnes :
    CSV      = valeur lue directement dans le CSV brut.
    RECALC   = valeur recalculée par src/url_features.py depuis la chaîne URL.
               Les définitions exactes sont documentées dans url_features.py.
               Les 12 variables marquées RECALC ont un taux d'accord >= 98.5 %
               avec le CSV (voir reports/tables/url_features_check.csv).
               Les 6 variables marquées CSV_ONLY ont une définition inconnue ou
               non reproductible depuis l'URL seule.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Racine du projet (répertoire contenant src/)
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────
# Chemins des données
# ─────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Nom du fichier CSV brut (à placer dans data/raw/)
RAW_FILENAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
RAW_CSV = RAW_DIR / RAW_FILENAME

# Fichiers intermédiaires
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# ─────────────────────────────────────────────
# Chemins des modèles
# ─────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ─────────────────────────────────────────────
# Chemins des rapports
# ─────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# ─────────────────────────────────────────────
# Paramètres du dataset
# ─────────────────────────────────────────────

# Colonne cible originale dans le CSV
ORIGINAL_TARGET_COL = "label"

# Nom de la cible recodée (phishing = 1, légitime = 0)
# Recodage : phishing = 1 - label
TARGET_COL = "phishing"

# Colonnes identifiants / texte à exclure de X
EXCLUDE_COLS = ["FILENAME", "URL", "Domain", "Host", "TLD", "Title", "row_id",
                ORIGINAL_TARGET_COL, TARGET_COL]

# Colonne identifiant pour le regroupement (GroupKFold / GroupShuffleSplit)
GROUP_COL = "Host"

# ─────────────────────────────────────────────
# Reproductibilité
# ─────────────────────────────────────────────
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# Validation croisée
# ─────────────────────────────────────────────
N_FOLDS = 5          # K pour GroupKFold
TEST_SIZE = 0.20     # proportion du jeu de test

# ─────────────────────────────────────────────
# Familles de variables numériques
# (les 50 colonnes numériques hors label)
# ─────────────────────────────────────────────

# 18 variables de la famille URL_ONLY (telles que définies dans le CSV original)
URL_ONLY_FEATURES = [
    "URLLength",              # source : CSV (définition non reproductible depuis URL)
    "DomainLength",           # source : RECALC (accord 99.98 %)
    "IsDomainIP",             # source : RECALC (accord 99.98 %)
    "TLDLength",              # source : RECALC (accord 99.74 %)
    "NoOfSubDomain",          # source : RECALC (accord 99.74 %)
    "HasObfuscation",         # source : RECALC (accord 99.84 %)
    "NoOfObfuscatedChar",     # source : RECALC (accord 99.67 %)
    "ObfuscationRatio",       # source : RECALC (accord 99.64 %)
    "NoOfLettersInURL",       # source : CSV_ONLY (définition inconnue, accord 0 %)
    "LetterRatioInURL",       # source : CSV_ONLY (dépend de NoOfLettersInURL, accord 0 %)
    "NoOfDegitsInURL",        # source : RECALC (accord 99.54 %)
    "DegitRatioInURL",        # source : CSV_ONLY (arrondi 3 décimales in CSV, accord 79.5 %)
    "NoOfEqualsInURL",        # source : RECALC (accord 99.89 %)
    "NoOfQMarkInURL",         # source : RECALC (accord 99.98 %)
    "NoOfAmpersandInURL",     # source : RECALC (accord 98.73 %, seuil 98.5 %)
    "NoOfOtherSpecialCharsInURL",  # source : CSV_ONLY (définition inconnue, accord 0 %)
    "SpacialCharRatioInURL",  # source : CSV_ONLY (définition inconnue, accord 0 %)
    "IsHTTPS",                # source : RECALC (accord 99.78 %)
]

# Origine de chaque colonne (pour référence)
# "RECALC"   : reproduit par src/url_features.py, taux >= 98.5 %
# "CSV_ONLY" : valeur originale du CSV uniquement (non reproductible)
COLUMN_SOURCES = {
    # ── URL_ONLY ──────────────────────────────────────────────────────────
    "URLLength":                    "CSV_ONLY",  # diff systematique de -1 sur legitimes
    "DomainLength":                 "RECALC",
    "IsDomainIP":                   "RECALC",
    "TLDLength":                    "RECALC",    # désaccords sur IPs phishing
    "NoOfSubDomain":                "RECALC",    # désaccords sur IPs phishing
    "HasObfuscation":               "RECALC",
    "NoOfObfuscatedChar":           "RECALC",
    "ObfuscationRatio":             "RECALC",
    "NoOfLettersInURL":             "CSV_ONLY",  # définition inconnue
    "LetterRatioInURL":             "CSV_ONLY",  # dépend de NoOfLettersInURL
    "NoOfDegitsInURL":              "RECALC",
    "DegitRatioInURL":              "CSV_ONLY",  # arrondi 3 déc. in CSV
    "NoOfEqualsInURL":              "RECALC",
    "NoOfQMarkInURL":               "RECALC",
    "NoOfAmpersandInURL":           "RECALC",    # hypothèse : & encodé différemment
    "NoOfOtherSpecialCharsInURL":   "CSV_ONLY",  # définition inconnue
    "SpacialCharRatioInURL":        "CSV_ONLY",  # définition inconnue
    "IsHTTPS":                      "RECALC",
    # ── PAGE_FEATURES ──────────────────────────────────────────────────────
    "LineOfCode":             "CSV",
    "LargestLineLength":       "CSV",
    "HasTitle":               "CSV",
    "HasFavicon":             "CSV",
    "Robots":                 "CSV",
    "IsResponsive":           "CSV",
    "NoOfURLRedirect":        "CSV",
    "NoOfSelfRedirect":       "CSV",
    "HasDescription":         "CSV",
    "NoOfPopup":              "CSV",
    "NoOfiFrame":             "CSV",
    "HasExternalFormSubmit":  "CSV",
    "HasSocialNet":           "CSV",
    "HasSubmitButton":        "CSV",
    "HasHiddenFields":        "CSV",
    "HasPasswordField":       "CSV",
    "Bank":                   "CSV",
    "Pay":                    "CSV",
    "Crypto":                 "CSV",
    "HasCopyrightInfo":       "CSV",
    "NoOfImage":              "CSV",
    "NoOfCSS":                "CSV",
    "NoOfJS":                 "CSV",
    "NoOfSelfRef":            "CSV",
    "NoOfEmptyRef":           "CSV",
    "NoOfExternalRef":        "CSV",
    # ── DERIVED_FEATURES ───────────────────────────────────────────────────
    "URLSimilarityIndex":     "CSV",
    "CharContinuationRate":   "CSV",
    "TLDLegitimateProb":      "CSV",
    "URLCharProb":            "CSV",
    "DomainTitleMatchScore":  "CSV",
    "URLTitleMatchScore":     "CSV",
}

# ─────────────────────────────────────────────
# OBSOLÈTE : C_VALIDATED_FEATURES (ancienne restriction à 12 features)
# L'expérience C utilise désormais l'ensemble des 18 variables URL_ONLY recalculées.
# ─────────────────────────────────────────────
C_VALIDATED_FEATURES = [
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "IsHTTPS",
]

# 6 variables URL_ONLY non reproductibles exclues de B'
C_EXCLUDED_FEATURES = [
    "URLLength",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "DegitRatioInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
]

# 26 variables nécessitant le contenu de la page
PAGE_FEATURES = [
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
]

# 6 variables dérivées (origine à auditer)
DERIVED_FEATURES = [
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
]

# Ensemble complet des 50 variables numériques (hors label)
ALL_NUMERIC_FEATURES = URL_ONLY_FEATURES + PAGE_FEATURES + DERIVED_FEATURES

# Assertions de cohérence (vérifiées à l'import)
assert len(URL_ONLY_FEATURES) == 18, f"URL_ONLY doit avoir 18 vars, a {len(URL_ONLY_FEATURES)}"
assert len(PAGE_FEATURES) == 26, f"PAGE doit avoir 26 vars, a {len(PAGE_FEATURES)}"
assert len(DERIVED_FEATURES) == 6, f"DERIVED doit avoir 6 vars, a {len(DERIVED_FEATURES)}"
assert len(ALL_NUMERIC_FEATURES) == 50, f"Total doit être 50, a {len(ALL_NUMERIC_FEATURES)}"

# Vérification d'absence de doublons dans les familles
_all_set = set(ALL_NUMERIC_FEATURES)
assert len(_all_set) == 50, "Des variables sont dupliquées entre familles !"

# ─────────────────────────────────────────────
# Paramètres de prétraitement TLD
# ─────────────────────────────────────────────
TOP_N_TLD = 20          # nombre de TLD à conserver explicitement
TLD_OTHER_LABEL = "autre"  # étiquette pour les TLD rares

# ─────────────────────────────────────────────
# Grilles d'hyperparamètres pour GridSearchCV
# (petites et raisonnables)
# ─────────────────────────────────────────────
PARAM_GRIDS = {
    "LogisticRegression": {
        "classifier__C": [0.1, 1.0, 10.0],
    },
    "DecisionTree": {
        "classifier__max_depth": [5, 10, 20, None],
    },
    "RandomForest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 20],
    },
    "KNN": {
        "classifier__n_neighbors": [5, 15],
    },
}

# Taille maximale du sous-échantillon pour KNN et SVC-RBF (lignes)
KNN_SUBSAMPLE_SIZE = 40_000
SVC_RBF_SUBSAMPLE_SIZE = 30_000

# Métrique de refit pour GridSearchCV
REFIT_METRIC = "average_precision"

# Métriques enregistrées pour GridSearchCV
SCORING_METRICS = ["average_precision", "roc_auc", "precision", "recall", "f1"]

# Fichier des features recalculées pour l'expérience C et D
URL_FEATURES_CSV = PROCESSED_DIR / "url_features_all.csv"
HOST_FEATURES_CSV = PROCESSED_DIR / "host_features_all.csv"

# 10 caractéristiques de l'hôte (Expérience D - HostNoOfOtherSpecial retiré car constante=0)
HOST_FEATURES = [
    "HostLength",
    "HostTLDLength",
    "HostNoOfSubDomain",
    "HostIsIP",
    "HostNoOfDigits",
    "HostDigitRatio",
    "HostNoOfLetters",
    "HostLetterRatio",
    "HostNoOfHyphens",
    "HostLongestLabel",
]

# ─────────────────────────────────────────────
# Définition des expériences
# ─────────────────────────────────────────────
# A  : 50 variables numériques du CSV brut (URL_ONLY + PAGE + DERIVED)
# B  : A moins DERIVED → 44 variables du CSV brut (URL_ONLY + PAGE)
# B' : B moins les 6 colonnes URL_ONLY non reproductibles → 38 variables du CSV brut
# C  : 18 variables URL_ONLY recalculées (source : url_features_all.csv)
# D  : 10 caractéristiques de l'hôte normalisé (source : host_features_all.csv)

# Variables de B = 44 variables (URL_ONLY + PAGE_FEATURES)
B_FEATURES = URL_ONLY_FEATURES + PAGE_FEATURES

# Variables de B' = B sans les 6 colonnes divergentes du CSV
_B_PRIME_EXCLUDED = set(C_EXCLUDED_FEATURES)
B_PRIME_FEATURES = [
    f for f in URL_ONLY_FEATURES + PAGE_FEATURES
    if f not in _B_PRIME_EXCLUDED
]  # 38 variables du CSV

EXPERIMENT_FEATURES = {
    "A":  ALL_NUMERIC_FEATURES,               # 50 vars (source : CSV brut)
    "B":  B_FEATURES,                         # 44 vars (source : CSV brut)
    "B'": B_PRIME_FEATURES,                   # 38 vars (source : CSV brut)
    "B_prime": B_PRIME_FEATURES,             # alias pour le système de fichiers
    "C":  URL_ONLY_FEATURES,                  # 18 vars (source : url_features_all.csv)
    "D":  HOST_FEATURES,                      # 10 vars (source : host_features_all.csv)
}

# Source de chaque colonne pour chaque expérience
FEATURE_SOURCES_BY_EXPERIMENT = {
    "A": {col: "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv" for col in EXPERIMENT_FEATURES["A"]},
    "B": {col: "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv" for col in EXPERIMENT_FEATURES["B"]},
    "B'": {col: "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv" for col in EXPERIMENT_FEATURES["B'"]},
    "B_prime": {col: "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv" for col in EXPERIMENT_FEATURES["B_prime"]},
    "C": {col: "data/processed/url_features_all.csv" for col in EXPERIMENT_FEATURES["C"]},
    "D": {col: "data/processed/host_features_all.csv" for col in EXPERIMENT_FEATURES["D"]},
}


def get_experiment_tag(experiment: str) -> str:
    """
    Retourne l'identifiant d'expérience pour le nommage des fichiers sur disque.
    Assainit 'B\'' en 'B_prime' pour éviter l'apostrophe dans les chemins.
    """
    if experiment in ("B'", "B_prime"):
        return "B_prime"
    return str(experiment)

# Longueur maximale acceptée pour une URL en prédiction
MAX_URL_LENGTH = 2048
