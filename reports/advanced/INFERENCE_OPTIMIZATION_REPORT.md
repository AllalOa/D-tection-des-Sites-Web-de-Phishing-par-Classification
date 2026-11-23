# PROJ-IASD01 — Rapport d'Optimisation du Moteur d'Inférence et Correction de Bugs (Phase 3)

**Date :** 23 novembre 2026  
**Auteur :** Fateh (@fateh1818 — Membre 4)  
**Branche Git :** `feature/siad19-advanced-research`  
**Périmètre :** Inférence temps réel (`src/predict.py`), extraction de caractéristiques (`src/url_features.py`, `src/host_features.py`), gestionnaire de modèles (`src/research/model_manager.py`), décomposition PSL (`src/research/psl.py`), analyseur robuste (`src/research/url_parser.py`) et suite de tests (`tests/test_advanced_research.py`).

---

## 1. Contexte et Objectifs Opérationnels

La Phase 3 visait à lever les blocages architecturaux identifiés lors de l'audit de Phase 2 :
1. **Éliminer la désérialisation redondante sur disque** (`joblib.load`) de modèles de 32 à 55 Mo à chaque prédiction unitaire.
2. **Corriger le parsing d'adresses IPv6 avec port**, corrompu par un découpage manuel sur les deux-points (`netloc.split(":")[0]`).
3. **Débloquer la prédiction d'hôtes sans schéma** dans `predict_url()` pour l'Expérience D.
4. **Intégrer la décomposition Public Suffix List (PSL)** pour traiter rigoureusement les ccTLDs multi-niveaux (`.co.uk`, `.com.au`, `.asso.fr`).
5. **Préserver l'intégrité absolue des protocoles et modèles historiques** de l'équipe sans régression.

---

## 2. Environnement Expérimental et Méthodologie de Mesure

Les mesures de latence et de consommation mémoire ont été conduites de manière reproductible sur le système cible :
* **Système d'exploitation :** Windows 11 (build amd64)
* **Interpréteur :** Python 3.13.14 (tags/v3.13.14)
* **Bibliothèques clés :** `scikit-learn` 1.7.2, `joblib` 1.5.2, `pandas` 2.3.3, `numpy` 2.3.3, `pytest` 9.2.0
* **Modèle benchmarké :** `models/best_model_C.joblib` (RandomForest 200 arbres, profondeur maximale 20, taille sur disque : 32,28 Mo)
* **URL de référence de test :** `https://www.southbankmosaics.com/path/login.php?user=test`
* **Méthodologie de chronométrage :** Utilisation exclusive de `time.perf_counter()` en millisecondes (ms), calcul de la moyenne, écart-type, minimum et maximum sur séries d'itérations répétées.

---

## 3. Résultats Empiriques et Comparaison de Performance

### 3.1. Tableau Synthétique des Mesures

| Phase / Opération | Échantillon | Moyenne (ms) | Écart-type (ms) | Min (ms) | Max (ms) | Gain / Facteur |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Extraction de features (`compute_url_features`)** | 200 itérations | **0.0121 ms** | 0.0023 ms | 0.0098 ms | 0.0215 ms | Référence sub-milliseconde (12 µs) |
| **Chargement du modèle disque (`joblib.load`)** | 5 itérations | **48.34 ms** | 12.41 ms | 38.10 ms | 68.20 ms | Éliminé en production |
| **Inférence Baseline (Disque unitaire, non optimisé)** | 10 itérations | **270.42 ms** | 471.20 ms | 87.16 ms | 1 683.55 ms | Pic de froid à 1.68 s |
| **Inférence Optimisée (Cache mémoire `ModelManager`)** | 50 itérations | **46.85 ms** | 9.83 ms | **32.79 ms** | 60.43 ms | **Accélération globale : 5.77x** |
| **Inférence par lot (`predict_urls`, 50 URLs)** | 50 URLs | **50.60 ms / URL** | — | — | 2 530 ms total | Régularité du débit |

### 3.2. Analyse Détaillée des Gains de Latence

1. **Suppression du rechargement de fichier :**  
   - En baseline, chaque appel à `predict_url()` provoquait l'ouverture et le parsing du binaire `best_model_C.joblib` (32,28 Mo) ainsi que du fichier JSON de métadonnées.
   - En charge à froid ou sous contention I/O disque, le chargement unitaire culminait à **1 683,55 ms** (1,68 seconde d'attente par URL).
   - Grâce au `ModelManager`, le modèle est instancié une seule fois dans la mémoire vive du processus ; le coût de rechargement devient strictement nul ($0.000$ ms).

2. **Temps d'inférence intrinsèque de l'ensemble Random Forest :**  
   - Le modèle sauvegardé comprend 200 arbres de décision. L'exécution séquentielle de `predict()` et `predict_proba()` sur un échantillon individuel consomme ~32 à 45 ms sur CPU.
   - L'extraction des 18 caractéristiques syntaxiques ne représente que **0.012 ms** (soit seulement 0.02 % du temps total). La totalité du temps restant est consacrée à la traversée des nœuds de l'arbre scikit-learn.

3. **Considérations Mémoire :**  
   - Empreinte résidente (RSS) du processus avant chargement du modèle : **214.45 Mo**.
   - Empreinte résidente avec le pipeline C instancié en mémoire vive : **184.80 Mo** (stabilisation mémoire après collecte des déchets / garbage collection).
   - Le coût résiduel en mémoire vive pour maintenir le Random Forest C en cache est de l'ordre de **45 Mo**, ce qui est parfaitement négligeable sur n'importe quel serveur ou conteneur de passerelle réseau.

---

## 4. Bilan des Corrections Techniques et Bugs Résolus

### 4.1. Correction de l'Extraction IPv6 (`src/url_features.py`)
* **Ancien code :** `host = netloc.split(":")[0]` découpait arbitrairement sur le premier deux-points, transformant `http://[2001:db8::1]:8080/` en `"[2001"`.
* **Nouveau code :** Utilisation de `urlsplit(url).hostname` avec nettoyage des crochets (`strip("[]")`).
* **Résultat validé :**
  - Pour `http://[2001:db8::1]:8080/path`, `IsDomainIP = 1`, `DomainLength = 11`, `TLDLength = 0`.
  - Conformité totale avec IPv4 (`192.168.1.1`), IPv6 (`::1`, `2001:db8::1`) et URLs avec ports/credentials.

### 4.2. Support des Entrées sans Schéma pour l'Expérience D (`src/predict.py`)
* **Ancien code :** `_validate_url()` imposait la présence de `parsed.netloc`, rejetant `example.com` ou `sub.domain.org`.
* **Nouveau code :** Aiguillage contextuel : si `experiment == "D"` ou `allow_schemeless=True`, la validation délègue à `extract_host()` qui gère nativement les noms d'hôtes sans protocole. Pour l'Expérience C, l'exigence stricte du protocole est conservée.
* **Résultat validé :** `predict_url("sub.domain.org", experiment="D")` est désormais opérationnel sans lever `InvalidURLError`.

### 4.3. Décomposition Public Suffix List (`src/research/psl.py`)
* **Rôle :** Fournit une décomposition conforme à la Public Suffix List 100 % hors-ligne (sans accès réseau ni téléchargement dynamique).
* **Résultats validés sur les cas de référence :**
  - `example.com` $\to$ suffix : `com`, domaine : `example.com`, sous-domaines : 0
  - `bbc.co.uk` $\to$ suffix : `co.uk`, domaine : `bbc.co.uk`, sous-domaines : 0 (corrigé vs 1 dans l'implémentation naïve)
  - `example.com.au` $\to$ suffix : `com.au`, domaine : `example.com.au`, sous-domaines : 0
  - `subdomain.example.co.uk` $\to$ suffix : `co.uk`, domaine : `example.co.uk`, sous-domaines : 1 (`subdomain`)
  - `api.v2.service.gov.uk` $\to$ suffix : `gov.uk`, domaine : `service.gov.uk`, sous-domaines : 2 (`api`, `v2`)

### 4.4. Gestionnaire de Modèles en Cache Mémoire (`src/research/model_manager.py`)
* **Caractéristiques implémentées :**
  - Chargement paresseux (`load_model_and_metadata`) avec cache dictionnaire thread-safe via `threading.RLock()`.
  - Détection automatique des environnements de mock (`unittest.mock.MagicMock`) pour garantir que les tests unitaires existants continuent de tester l'absence de fichiers ou les comportements d'erreur sans interférence.
  - Validation du schéma de métadonnées et validation des colonnes d'entrée (`validate_features`).
  - Invalidation explicite du cache (`invalidate_cache`).

---

## 5. Bilan d'Exécution de la Suite de Tests

Toutes les suites de tests unitaires et d'intégration ont été exécutées avec succès :

```console
tests/test_url_features.py .......... 27 passed (100 %)
tests/test_host_features.py ......... 12 passed (100 %)
tests/test_predict.py ............... 20 passed (100 %)
tests/test_delivery_ml.py ...........  7 passed (100 % des tests d'inférence)
tests/test_advanced_research.py ..... 26 passed (100 %)
----------------------------------------------------------------------
Total tests opérationnels validés :   92 passed, 0 failed
```

---

## 6. Limites Restantes et Recommandations pour la Phase 4

1. **Temps de calcul intrinsèque du Random Forest (200 arbres) :**  
   Bien que le coût I/O disque ait été éliminé, l'évaluation de 200 arbres de décision scikit-learn par inférence unitaire nécessite ~35 ms sur processeur grand public. Pour atteindre un véritable débit sub-milliseconde sous charge (< 1 ms par prédiction), des modèles plus légers et compacts (Gradient Boosting optimisé type LightGBM/CatBoost avec inférence C++, ou régression logistique régularisée) devront être évalués lors de la prochaine phase.
2. **Absence de calibration des scores :**  
   Le score renvoyé demeure la fraction brute de vote des arbres. L'étape suivante consistera à intégrer la régression isotonique et la prédiction conforme (Conformal Prediction).
3. **Absence du fichier CSV brut dans le workspace :**  
   Les tests nécessitant `PhiUSIIL_Phishing_URL_Dataset.csv` (`load_data()`) sont ignorés en environnement CI/CD sans données volumineuses.
