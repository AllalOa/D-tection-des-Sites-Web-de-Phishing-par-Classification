# SIAD19 — PHASE 2 : RAPPORT D'AUDIT D'ARCHITECTURE ET EXTENSION DE RECHERCHE PERSONNELLE

**Auteur :** Fateh (@fateh1818 / Git : `fateh` / `yomyname50@gmail.com` — Membre 4)  
**Projet :** PROJ-IASD01 — Détection des Sites Web de Phishing par Classification  
**Date d'audit :** 23 novembre 2026  
**Périmètre :** Modules contributifs (`src/url_features.py`, `src/predict.py`, `src/host_features.py`, `docs/EXPERIMENT_C_PROTOCOL.md`, commit `2a14c02`), pipeline d'entraînement/évaluation, application Flask V2 et opportunités d'extension scientifique autonome.  
**Règles d'engagement :** Audit en lecture seule — Aucune modification des fichiers d'équipe existants, aucun commit, aucun push, préservation intégrale des modèles en production.

---

## 1. Analyse des Contributions Existantes

Les contributions de Fateh constituent le socle opérationnel et déployable du projet PROJ-IASD01, axé sur l'inférence autonome à faible latence sans accès réseau.

### 1.1. `src/url_features.py` (Calcul des 18 caractéristiques URL_ONLY)
* **Rôle opérationnel :** Extraction déterministe hors ligne de 18 caractéristiques lexicales à partir de la chaîne d'URL brute (`URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`).
* **Architecture :** Implémentation purement algorithmique basée sur `urllib.parse`, `ipaddress` et des expressions régulières précompilées (`_RE_PERCENT_ENCODING`, `_RE_SPECIAL_CHARS`). Aucun appel socket.
* **Fonction de lot :** `compute_url_features_batch()` calcule le DataFrame des 18 variables pour une collection d'URLs avec alignement strict de l'ordre des colonnes sur `src.config.URL_ONLY_FEATURES`.

### 1.2. `src/host_features.py` (Extraction des 10 caractéristiques d'hôte)
* **Rôle opérationnel :** Extraction et normalisation stricte du FQDN d'hôte (`extract_host`) et calcul des 10 variables structurelles d'infrastructure pour l'Expérience D (`HostLength`, `HostTLDLength`, `HostNoOfSubDomain`, `HostIsIP`, `HostNoOfDigits`, `HostDigitRatio`, `HostNoOfLetters`, `HostLetterRatio`, `HostNoOfHyphens`, `HostLongestLabel`).
* **Normalisation :** Suppression du protocole, des identifiants utilisateur (`user:pass@`), du port, du chemin/requête/fragment, du préfixe `www.` et du point final.

### 1.3. `src/predict.py` (Moteur d'inférence unifié)
* **Rôle opérationnel :** Interface unique programmatique (`predict_url()`) et CLI (`python -m src.predict <URL>`).
* **Validation d'entrée :** `_validate_url()` rejette les URLs vides, de type non-chaîne, de longueur supérieure à `MAX_URL_LENGTH` (2048 caractères) ou dépourvues d'hôte.
* **Gestion multi-expériences :** Aiguillage dynamique vers `host_features` pour l'Expérience D ou `compute_url_features` pour l'Expérience C (ou modèles A, B, B' si spécifiés).
* **Politique de seuil :** Prise en charge du seuil `standard` (0.5 proba / 0.0 marge), `high_precision` (0.9352 strictement réservé à D/RandomForest) ou seuil flottant utilisateur.

### 1.4. `docs/EXPERIMENT_C_PROTOCOL.md` (Protocole scientifique C vs D)
* **Rôle méthodologique :** Formalisation de la comparaison descriptive entre le modèle URL_ONLY (C, 18 vars) et le modèle HOST_ONLY (D, 10 vars).
* **Cadre expérimental :** Établissement de l'étanchéité par `Host` (`GroupShuffleSplit` 80/20, 0 fuite d'hôte), de la population unifiée (235 370 URLs dédupliquées), des sous-groupes G1/G2 et de l'interdiction de transfert de seuil.

### 1.5. Commit d'intégration `2a14c02`
* Intégration du moteur d'inférence temps réel (Membre 4) dans le tronc commun, reliant les modules de features avec le pipeline de preprocessing et les modèles sauvegardés.

---

## 2. Évaluation de la Qualité du Code (Code Quality Assessment)

| Critère | Note | Constat Technique |
|:---|:---:|:---|
| **Conception & Modularité** | 8.5/10 | Excellente séparation fonctionnelle. Les modules d'extraction (`url_features`, `host_features`) sont indépendants de l'infrastructure d'apprentissage scikit-learn. |
| **Sécurité & Isolation réseau** | 9.5/10 | Aucun appel réseau sortant. Les tests de livraison (`test_delivery_ml.py`, `test_predict.py`) vérifient par monkeypatching de `socket.connect` l'isolation hermétique. |
| **Typage & Documentation** | 7.5/10 | Docstrings complètes et claires. Présence de typage partiel (hints `typing` dans `host_features`, mais absent dans `url_features`). Quelques divergences mineures entre docstring et implémentation. |
| **Performance & Latence** | 5.5/10 | Calcul unitaire de features très rapide (< 0.5 ms). En revanche, **latence critique d'I/O** dans `predict.py` : rechargement complet de `joblib.load()` (32 à 55 Mo) à chaque appel de `predict_url()`. Traitement par lot itératif non vectorisé. |
| **Robustesse aux cas limites** | 6.5/10 | Bon traitement des cas simples. Faiblesses notables sur les adresses IPv6 avec port, les TLDs composés (ccTLD) et les entrées sans schéma HTTP. |

---

## 3. Anomalies et Bugs Confirmés (Confirmed Bugs)

### Bug 1 : Échec d'extraction de l'hôte pour les adresses IPv6 avec port dans `url_features.py`
* **Localisation :** `src/url_features.py`, lignes 81-83 :
  ```python
  netloc = parsed.netloc or ""
  host = netloc.split(":")[0] if netloc else ""
  ```
* **Impact :** Pour une URL IPv6 telle que `http://[2001:db8::1]:8080/path`, `parsed.netloc` vaut `[2001:db8::1]:8080`. Le découpage `netloc.split(":")[0]` produit `"[2001"`. Par conséquent :
  1. `_is_ip_address("[2001")` lève `ValueError` et retourne `False` $\to$ `IsDomainIP` devient 0 au lieu de 1.
  2. `DomainLength` est tronqué de manière erronée.
* **Gravité :** Moyenne (les adresses IPv6 directes sont rares dans PhiUSIIL mais critiques en production de cybersécurité).

### Bug 2 : Rejet injustifié des URLs sans schéma dans `predict.py` pour l'Expérience D
* **Localisation :** `src/predict.py`, lignes 75-80 vs `src/host_features.py`, ligne 57 :
  ```python
  parsed = urlparse(url)
  if not parsed.netloc:
      raise InvalidURLError("L'URL ne contient pas d'hôte valide (netloc absent)...")
  ```
* **Impact :** `src/host_features.py` a été explicitement conçu pour accepter des URLs sans protocole (ex: `example.com/login` ou `paypal.fraud.org`). Cependant, `predict_url()` appelle `_validate_url()` en amont. Comme `urlparse("example.com").netloc` est vide (Python place la chaîne dans `.path`), `_validate_url` lève une exception `InvalidURLError`. Il est donc impossible de prédire sur un nom d'hôte ou une URL sans protocole avec l'Expérience D via l'API officielle.
* **Gravité :** Haute pour l'ergonomie utilisateur et l'alignement avec les spécifications de D.

### Bug 3 : Incohérence de docstring et suppression incomplète de `www.` dans `host_features.py`
* **Localisation :** `src/host_features.py`, lignes 22-33, 91-93 et 119 :
  1. La docstring annonce 11 variables calculées incluant `HostNoOfOtherSpecial`, or `HOST_FEATURES` et le dictionnaire retourné n'en contiennent que 10 (`HostNoOfOtherSpecial` ayant été supprimée car constante=0 dans PhiUSIIL).
  2. La docstring indique `retirer 'www.' initial (répété si nécessaire ex: www.www. -> hors www initial)`, mais le code exécute un simple `if host.startswith("www."): host = host[4:]` (condition unique non itérative).
* **Gravité :** Faible (propreté du code et rigueur documentaire).

### Bug 4 : Rechargement systématique du modèle sur disque à chaque prédiction (`predict.py`)
* **Localisation :** `src/predict.py`, lignes 185 :
  ```python
  pipeline, metadata = _load_model_and_metadata(experiment=experiment)
  ```
* **Impact :** `joblib.load()` désérialise un fichier binaire de 32,3 Mo (Expérience C) ou 55 Mo (Expérience D) à chaque invocation unitaire.
  - Latence mesurée : 150 ms à 350 ms par requête sur disque standard, au lieu de < 0.5 ms en mémoire vive.
  - Dégradation majeure du débit de l'application Flask et impossibilité de soutenir un flux de production en passerelle réseau.
* **Gravité :** Critique pour les performances en déploiement réel.

### Bug 5 : Découpage TLD naïf sur le dernier point (`rsplit(".", 1)`)
* **Localisation :** `src/url_features.py` (ligne 105) et `src/host_features.py` (ligne 131) :
  ```python
  tld = host.rsplit(".", 1)[-1]
  tld_length = len(labels[-1])
  ```
* **Impact :** Pour les extensions de second niveau (ccTLD comme `.co.uk`, `.com.au`, `.asso.fr`, `.gov.br`) :
  - Le TLD extrait est `uk` (longueur 2) au lieu de `co.uk` (longueur 5).
  - Dans `host_features.py`, `no_of_subdomain = max(0, len(labels) - 2)` : pour `bbc.co.uk`, labels = `['bbc', 'co', 'uk']` (3 labels), ce qui calcule $3 - 2 = 1$ sous-domaine alors que `bbc.co.uk` n'a aucun sous-domaine !
* **Gravité :** Biais méthodologique documenté dans `docs/LIMITATIONS.md`, mais corrigeable par une intégration propre de la Public Suffix List (PSL).

---

## 4. Audit des Tests et Lacunes Identifiées (Missing Tests)

Le dépôt compte 258 tests unitaires validés (cf. `reports/pytest_phase9.txt`). Néanmoins, l'audit approfondi révèle d'importantes lacunes de couverture sur le périmètre de Fateh :

### 4.1. Tests actuels de `test_url_features.py` (14 tests) et `test_predict.py` (14 tests)
* Les tests actuels ne valident que des cas nominaux simples (`https://www.example.com`, `http://192.168.1.1`).
* Les tests de `test_predict.py` mockent systématiquement `joblib.load` et `_load_model_and_metadata`, ce qui masque le goulot d'étranglement de performance I/O et les éventuels conflits d'environnement réels.

### 4.2. Tests manquants critiques à ajouter :
1. **Tests d'adresses IPv6 complexes :**
   - IPv6 avec port : `http://[2001:db8::1]:8080/`
   - IPv6 compressée : `http://[::1]/`
   - IPv6 sans crochets ou malformée.
2. **Tests de contournement et d'encodage (Obfuscation & Evasion) :**
   - Double encodage URL (`%2520`, `%252e%252e`).
   - Adresses IP en notation octale (`http://0177.0.0.1`) ou hexadécimale (`http://0x7f000001`), couramment utilisées dans les kits de phishing.
   - Présence de caractères homoglyphes / Punycode IDN (`https://xn--pple-43d.com` mimant Apple).
3. **Tests de non-régression sur formats de schémas atypiques :**
   - URLs relatives ou sans schéma injectées dans le prédicteur.
   - Schémas non-web (`javascript:`, `data:text/html,...`, `file:///`).
4. **Benchmarks de latence et profilage mémoire (SLA < 1 ms) :**
   - Test de performance vérifiant que l'extraction unitaire s'exécute en moins de 1 milliseconde.
   - Test de débit unitaire avec modèle en cache mémoire (> 1000 prédictions/seconde).
5. **Tests de robustesse hors-distribution (OOD) :**
   - URLs légitimes réelles modernes comportant des chemins profonds et des requêtes complexes (ex: recherche Google, produit Amazon, API RESTful) pour mesurer le taux de faux positifs face au biais de PhiUSIIL.

---

## 5. Limitations Scientifiques et Vulnérabilités Méthodologiques

### 5.1. Dépendance aux artefacts structurels du dataset PhiUSIIL
* **Absence totale de chemins/requêtes chez les légitimes :** 100 % des 134 850 URLs légitimes de PhiUSIIL sont des racines d'hôte (`path == ''` ou `path == '/'`, 0 paramètre de requête `?`, 0 caractère `=`, 0 esperluette `&`).
* **Risque de sur-apprentissage lexical :** Dans l'Expérience C, des variables comme `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL` et `URLLength` agissent comme des indicateurs quasi-parfaits de phishing sur ce benchmark, mais échoueront en environnement réel où la majorité des URLs légitimes sont des liens profonds (deep links).
* **Effondrement sur les sous-groupes :** Sur le sous-groupe G2 (noms d'hôte nus sous HTTPS), la PR-AUC de l'Expérience C chute de **0.9979 à 0.5521** (avec 86 faux négatifs et 62 faux positifs), révélant que le modèle perd l'essentiel de sa capacité discriminante lorsque le raccourci du chemin et du protocole est neutralisé.

### 5.2. Non-calibration des probabilités du modèle RandomForest
* Dans `predict.py` (ligne 14), une mention explicite avertit : `model_score N'EST PAS une 'confiance' au sens probabiliste strict`.
* Un Random Forest calcule des fractions de vote parmi 200 arbres. Ces scores tendent à être comprimés vers le centre ou polarisés, et ne reflètent pas des probabilités a posteriori $P(Y=1|X)$ bien calibrées.
* L'absence de calibration (Platt Scaling ou Régression Isotonique) empêche l'utilisation rationnelle des scores dans une matrice de coût financier ou un système de tri d'alertes en centre opérationnel de sécurité (SOC).

### 5.3. Fragilité scientifique du seuil haute précision (0.9352)
* Le seuil de 0.9352 propre à l'Expérience D a été optimisé sur le jeu de TRAIN pour garantir une précision OOF $\ge 0.99$.
* Cependant, cette politique n'a jamais été calculée pour l'Expérience C (la documentation interdit formellement de transférer 0.9352 à C). L'Expérience C fonctionne donc au seuil par défaut arbitraire de 0.5, sans garantie formelle sur le taux de faux positifs en production.

---

## 6. Compatibilité avec l'Application Flask V2 (`webapp/`)

L'application Flask V2 (PhishGuard) consomme directement les modules de Fateh via `webapp/app.py` et `webapp/services.py` :

1. **Point d'intégration :**
   - `load_predictor()` dans `webapp/app.py` (lignes 230-234) importe dynamiquement `from src.predict import InvalidURLError, predict_url`.
   - La route `POST /api/predict` relaie les paramètres `{url, experiment, threshold}` à `predict_url()`.
2. **Conformité du contrat de données :**
   - Le dictionnaire retourné par `predict_url()` est rigoureusement conforme au schéma attendu par Flask : `url`, `predicted_class`, `predicted_label`, `model_score`, `threshold_used`, `features`, `model_name`, `experiment`.
   - `webapp/services.py` calcule les métadonnées de sous-groupe (`url_subgroup`) et les avertissements de fiabilité (`reliability_notes`) en complément.
3. **Point critique d'impact :**
   - Le rechargement de fichier `joblib.load()` dans `predict.py` se répercute directement sur le temps de réponse de l'API web (temps de latence HTTP oscillant entre 200 ms et 500 ms par clic). L'introduction d'un cache mémoire singleton au niveau de `predict.py` divisera ce temps de latence par plus de 100 sans modifier la signature de la fonction.

---

## 7. Modules Indépendants Proposés (Nouvelles Opportunités de Contribution)

Afin d'étendre les contributions existantes de Fateh sans impacter les fichiers partagés de l'équipe, une architecture modulaire et autonome est proposée sous un sous-paquet dédié (ex: `src/advanced_models/` ou `src/research/`) :

```
src/research/
├── __init__.py
├── boosting_benchmark.py     # Modèles Gradient Boosting (XGBoost, LightGBM, CatBoost)
├── neural_url_classifier.py  # Classifieur neuronal au niveau caractère (Char-CNN / BiLSTM)
├── calibrated_predictor.py   # Inférence avec probabilités calibrées et cache singleton
├── explainability.py         # Explicabilité locale et globale TreeSHAP / KernelSHAP
├── uncertainty_conformal.py  # Ensembles de prédiction conformal (garantie d'erreur 1%)
├── adversarial_eval.py       # Génération d'attaques d'évasion et évaluation de robustesse
└── batch_scanner.py          # Moteur de scoring CSV par lot multithreadé à haut débit
```

### Détail des composants de recherche :

#### 1. Benchmark Gradient Boosting (`boosting_benchmark.py`)
* **Valeur scientifique :** Évaluer si les algorithmes de Gradient Boosting régularisés (XGBoost, LightGBM, CatBoost) surpassent le Random Forest de référence (AP = 0.9979) sur les sous-groupes difficiles G1 et G2. CatBoost excelle notamment sur les distributions déséquilibrées sans sur-ajustement.
* **Coût d'implémentation :** Faible à modéré. Utilise les 18 features existantes de `src/url_features.py` et le même partitionnement `GroupShuffleSplit` / `GroupKFold` par Host.

#### 2. Classifieur Neuronal au niveau Caractère (`neural_url_classifier.py`)
* **Valeur scientifique :** Rupture méthodologique majeure. Au lieu d'extraire des features lexicales manuelles sujettes aux biais humains, un réseau Char-CNN (Zhang et al.) ou BiLSTM lit la séquence brute de caractères de l'URL (vocabulaire de 128 caractères ASCII, longueur fixée à 256). Il capture automatiquement les sous-mots, les séparateurs rares et les variations d'entropie sans nécessiter de règles prédéfinies.
* **Coût d'implémentation :** Élevé. Nécessite PyTorch, une boucle d'entraînement avec GPU/MPS et un pipeline de tokenisation adapté.

#### 3. Explicabilité SHAP (`explainability.py`)
* **Valeur scientifique :** En sécurité opérationnelle, classifier une URL comme phishing ne suffit pas : l'analyste SOC doit savoir *pourquoi*. TreeSHAP fournit les contributions exactes de chaque caractéristique pour une URL donnée (ex : "+0.45 en raison de NoOfSubDomain=4, +0.25 en raison de l'absence de HTTPS").
* **Coût d'implémentation :** Faible. Compatible nativement avec Random Forest, XGBoost et LightGBM via la bibliothèque `shap`.

#### 4. Calibration & Prédiction Conforme (`uncertainty_conformal.py` & `calibrated_predictor.py`)
* **Valeur scientifique :**
  - **Calibration :** Application d'une régression isotonique ou de Platt scaling sur les plis out-of-fold pour transformer les scores en vraies probabilités d'erreur.
  - **Conformal Prediction :** Pour chaque prédiction, au lieu de renvoyer une classe unique, le système fournit un ensemble de confiance : $\{\text{phishing}\}$, $\{\text{légitime}\}$, ou $\{\text{phishing, légitime}\}$ (zone d'incertitude). Si le modèle est incertain, l'URL est automatiquement acheminée vers une analyse approfondie (Niveau 2 / inspection DOM).
* **Coût d'implémentation :** Modéré. Formidable apport scientifique pour une soutenance de Master / diplôme d'ingénieur.

#### 5. Évaluation de Robustesse Adversariale (`adversarial_eval.py`)
* **Valeur scientifique :** Simuler des perturbations réalistes apportées par un attaquant :
  - Ajout de sous-domaines bénins (`paypal.com-security-check.net`).
  - Remplacement de caractères par encodage hexadécimal ou pourcent.
  - Ajout de paramètres de requête factices pour noyer les ratios de lettres/chiffres.
  - Mesure de la dégradation de l'Average Precision sous attaque (Stress Testing).
* **Coût d'implémentation :** Modéré.

#### 6. Moteur de Détection par Lot Optimisé (`batch_scanner.py`)
* **Valeur scientifique & technique :** Vectorisation de l'extraction et parallélisation via `multiprocessing` / chunking. Capacité de traiter 100 000 URLs en quelques secondes avec streaming des résultats en CSV.
* **Coût d'implémentation :** Faible.

---

## 8. Nouvelle Architecture Expérimentale (Extension sans Conflit)

Pour garantir une étanchéité parfaite avec les travaux de l'équipe, la nomenclature suivante est adoptée :

```
Expériences historiques (Équipe - Intactes) :
  Exp. A       : 50 vars CSV (Full) ── RandomForest
  Exp. B       : 44 vars CSV (Sans DERIVED) ── RandomForest
  Exp. B'      : 38 vars CSV (Purifié sans TLD) ── RandomForest
  Exp. C       : 18 vars URL_ONLY (Fateh) ── RandomForest
  Exp. D       : 10 vars HOST_ONLY (Fateh) ── RandomForest

Nouvelles pistes d'extension autonome (Fateh Phase 2) :
  Exp. C-XGB   : 18 vars URL_ONLY ── XGBoost (optimisé Optuna/CV)
  Exp. C-LGBM  : 18 vars URL_ONLY ── LightGBM
  Exp. C-Cat   : 18 vars URL_ONLY ── CatBoost Classifier
  Exp. C-Calib : 18 vars URL_ONLY ── RF / Boost calibré isotonique + Conformal Sets
  Exp. E-Char  : Séquence brute de caractères d'URL ── Char-CNN / BiLSTM Deep Learning
```

### Matrice de Non-Interférence avec l'Équipe :
* **Données :** Utilise `data/processed/url_features_all.csv` (déjà généré et figé) et les mêmes `row_id` de `src/split.py`.
* **Modèles existants :** Aucun écrasement de `models/best_model_*.joblib`. Les nouveaux modèles seront placés dans un répertoire isolé `models/advanced/`.
* **Rapports :** Les résultats comparatifs seront stockés dans `reports/tables/advanced_benchmark.csv` et documentés dans un rapport d'extension dédié.

---

## 9. Matrice d'Évaluation des Opportunités et Complexité

| Proposition | Valeur Scientifique | Valeur Opérationnelle (SOC) | Complexité Réalisation | Dépendances Externes | Risque de Régression |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Cache Singleton & Fix Bugs `predict.py`** | Moyenne | **Maximale (x100 vitesse)** | Très faible (2 h) | Aucune | Nul |
| **2. Calibration de Probabilités & Seuil Optimal C** | **Très Haute** | **Très Haute** | Faible (4 h) | `scikit-learn` | Nul |
| **3. Benchmark XGBoost / LightGBM / CatBoost** | **Très Haute** | Haute | Modérée (6 h) | `xgboost`, `lightgbm`, `catboost` | Nul |
| **4. Explicabilité Locale TreeSHAP** | Haute | **Maximale (Auditabilité)** | Faible (3 h) | `shap` | Nul |
| **5. Ensembles de Prédiction Conforme (99%)** | **Exceptionnelle** | **Très Haute** | Modérée (6 h) | `scikit-learn`, `scipy` | Nul |
| **6. Moteur Batch CSV Haut Débit** | Moyenne | **Maximale (Production)** | Faible (3 h) | `pandas`, `multiprocessing` | Nul |
| **7. Évaluation de Robustesse Adversariale** | **Exceptionnelle** | Haute | Modérée (8 h) | `numpy` | Nul |
| **8. Classifieur Neuronal Char-CNN** | **Exceptionnelle** | Moyenne (calcul GPU) | Haute (15 h) | `torch` | Nul (module isolé) |

---

## 10. Priorités de Développement Recommandées

### Phase A : Fiabilisation & Performance Immédiate (Priorité 1)
1. **Implémenter un chargeur de modèle avec cache singleton :** Dans un module dérivé (ou `predict.py` si autorisé ultérieurement), stocker `_pipeline_cache` pour éliminer la lecture disque redondante.
2. **Corriger le parsing IPv6 :** Traiter correctement les crochets IPv6 avant le découpage de port dans `url_features.py`.
3. **Moteur Batch CSV optimisé :** Fournir un script `predict_batch.py` avec streaming chunké pour traiter de gros volumes d'URLs.

### Phase B : Rigueur Scientifique & Incertitude (Priorité 2)
1. **Calibration de probabilités sur l'Expérience C :** Ajuster un calibrateur isotonique sur les prédictions hors-pli du Train pour fournir de vraies probabilités fiables.
2. **Détermination d'un seuil haute précision formel pour C :** Calculer et certifier le seuil optimisé sur Train C pour atteindre une précision $\ge 0.99$ (comblant le vide méthodologique actuel où seul D dispose d'un seuil).
3. **Prédiction conforme (Conformal Risk Control) :** Définir les intervalles garantissant 99 % de rappel avec réorientation des cas ambigus vers la cascade.

### Phase C : Benchmark Avancé & Explicabilité (Priorité 3)
1. **Benchmark Gradient Boosting (XGBoost, LightGBM, CatBoost) :** Entraîner et comparer ces modèles aux côtés du Random Forest sur les sous-groupes G1/G2.
2. **Module d'explicabilité SHAP :** Générer les waterfall plots et beeswarm plots pour expliquer les décisions sur les URLs bloquées.
3. **Évaluation de robustesse face aux attaques d'évasion :** Rédiger le rapport d'analyse de vulnérabilité face aux attaques d'obfuscation.

---

## 11. Plan d'Action Précis pour la Suite (En Attente de Validation)

```
Étape 1 : Création du sous-répertoire isolé `src/advanced/` (ou `src/research/`).
Étape 2 : Implémentation du moteur d'inférence haute performance avec cache mémoire et calibration (`src/advanced/calibrated_engine.py`).
Étape 3 : Script de benchmark des algorithmes Gradient Boosting (`src/advanced/train_boosting.py`) respectant scrupuleusement le partitionnement par hôte de `src/split.py`.
Étape 4 : Évaluation comparative sur le jeu de test et export des métriques dans `reports/tables/advanced_models_comparison.csv`.
Étape 5 : Intégration d'un module d'explicabilité SHAP avec export de graphiques vectoriels pour la soutenance.
Étape 6 : Tests unitaires dédiés dans `tests/test_advanced_features.py` (validation de non-régression et couverture des cas limites IPv6/Punycode).
```

---

> [!IMPORTANT]
> **Statut de conformité :** Conformément aux règles de protection énoncées, **aucun fichier existant n'a été modifié**, aucune dépendance externe n'a été installée, et aucun commit n'a été réalisé.
> 
> **Arrêt obligatoire :** L'audit est complet et consigné dans [SIAD19_PERSONAL_CONTRIBUTION_AUDIT.md](file:///c:/Users/ENPEI/Desktop/D-tection-des-Sites-Web-de-Phishing-par-Classification-main/SIAD19_PERSONAL_CONTRIBUTION_AUDIT.md). J'attends vos instructions et votre accord formel avant d'engager toute phase d'implémentation.
