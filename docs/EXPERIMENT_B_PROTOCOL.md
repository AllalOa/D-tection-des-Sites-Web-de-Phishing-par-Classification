# PROTOCOLE OPÉRATOIRE SCIENTIFIQUE — EXPÉRIENCE B (CONTENT_FREE / 44 FEATURES + TLD)

**Projet :** PROJ-IASD01 — Détection des Sites Web de Phishing par Classification  
**Auteur :** Machine Learning Senior Engineer  
**Date de révision :** 2026-10-30 (Phase 6c-bis)  
**Statut :** Préparation, audit et vérification statique finalisés — En attente d'autorisation formelle d'entraînement  

---

## 1. Objectif Scientifique de l'Expérience B

L'expérience B vise à évaluer la capacité prédictive d'un système de détection fondé sur les **caractéristiques intrinsèques de l'URL et du contenu de page HTML/JS**, en **excluant rigoureusement les 6 caractéristiques dérivées/heuristiques** (`DERIVED_FEATURES`).

### 1.1. Motivation scientifique de l'exclusion des DERIVED_FEATURES
Dans l'Expérience A (FULL FEATURES, 50 variables + TLD), le modèle baseline `DecisionTree(max_depth=1)` sur la seule variable `URLSimilarityIndex` a obtenu une Average Precision (AP) de **0.9956**, et le `RandomForest` final a atteint une AP parfaite de **1.0000** (0 faux positif, 0 faux négatif sur le Test).

> [!IMPORTANT]
> **Audit du mécanisme de `URLSimilarityIndex` :**  
> Le mécanisme exact de calcul de `URLSimilarityIndex` n'a pas pu être établi à partir des sources locales disponibles.  
> Les 6 variables dérivées (`URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `DomainTitleMatchScore`, `URLTitleMatchScore`) sont des caractéristiques prétraitées fournies directement dans le jeu de données PhiUSIIL.  
> Leur calcul sur une nouvelle URL inconnue nécessiterait des modèles ou tables de probabilités externes non documentés localement.

L'Expérience B permet ainsi de répondre à la question de recherche fondamentale :
> *Quelle est la performance réelle d'un classifieur exploitant les caractéristiques observables d'URL et de page sans bénéficier de ces signaux dérivés précalculés ?*

---

## 2. Inventaire Exhaustif des 44 Caractéristiques Numériques + TLD (45 Colonnes d'Entrée)

Le pipeline de l'Expérience B reçoit en entrée **45 colonnes au total** avant prétraitement :
- **44 caractéristiques numériques originales** issues de `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (`EXPERIMENT_FEATURES["B"]`).
- **1 caractéristique catégorielle** : `TLD` (extension de premier niveau).

### 2.1. Les 18 Caractéristiques d'URL (`URL_ONLY_FEATURES`)
Colonnes issues du fichier CSV brut de PhiUSIIL :
1. `URLLength` : Longueur totale de l'URL brute.
2. `DomainLength` : Longueur du domaine.
3. `IsDomainIP` : Présence d'une adresse IP dans le domaine (0 ou 1).
4. `TLDLength` : Longueur du TLD.
5. `NoOfSubDomain` : Nombre de sous-domaines (définition originale du CSV).
6. `HasObfuscation` : Présence de caractères encodés (%) ou obfusqués (0 ou 1).
7. `NoOfObfuscatedChar` : Nombre total de caractères encodés.
8. `ObfuscationRatio` : Ratio de caractères obfusqués sur la longueur totale.
9. `NoOfLettersInURL` : Nombre de lettres alphabétiques (définition CSV originale).
10. `LetterRatioInURL` : Ratio de lettres dans l'URL.
11. `NoOfDegitsInURL` : Nombre de chiffres dans l'URL.
12. `DegitRatioInURL` : Ratio de chiffres dans l'URL.
13. `NoOfEqualsInURL` : Nombre de caractères `=`.
14. `NoOfQMarkInURL` : Nombre de caractères `?`.
15. `NoOfAmpersandInURL` : Nombre de caractères `&`.
16. `NoOfOtherSpecialCharsInURL` : Nombre d'autres caractères spéciaux.
17. `SpacialCharRatioInURL` : Ratio d'autres caractères spéciaux.
18. `IsHTTPS` : Protocole HTTPS présent (0 ou 1).

### 2.2. Les 26 Caractéristiques de Contenu de Page (`PAGE_FEATURES`)
Liste de référence unique extraite directement de `src/config.py` :
1. `LineOfCode` : Nombre total de lignes de code HTML/JS.
2. `LargestLineLength` : Longueur maximale d'une ligne de code.
3. `HasTitle` : Présence d'un titre de page (0 ou 1).
4. `HasFavicon` : Présence d'une icône favicon (0 ou 1).
5. `Robots` : Détection d'un fichier ou métatag robots (0 ou 1).
6. `IsResponsive` : Conception réactive/adaptative pour mobile (0 ou 1).
7. `NoOfURLRedirect` : Nombre de redirections URL.
8. `NoOfSelfRedirect` : Nombre de redirections internes vers la même page.
9. `HasDescription` : Présence d'une balise meta description (0 ou 1).
10. `NoOfPopup` : Nombre de fenêtres popup détectées.
11. `NoOfiFrame` : Nombre d'iframes intégrées.
12. `HasExternalFormSubmit` : Formulaire soumettant des données vers un domaine tiers (0 ou 1).
13. `HasSocialNet` : Liens vers des réseaux sociaux (0 ou 1).
14. `HasSubmitButton` : Présence d'un bouton de soumission de formulaire (0 ou 1).
15. `HasHiddenFields` : Présence de champs de saisie masqués (0 ou 1).
16. `HasPasswordField` : Présence d'un champ de mot de passe (0 ou 1).
17. `Bank` : Présence de mots-clés bancaires (0 ou 1).
18. `Pay` : Présence de mots-clés liés au paiement (0 ou 1).
19. `Crypto` : Présence de mots-clés liés aux cryptomonnaies (0 ou 1).
20. `HasCopyrightInfo` : Présence d'une mention de copyright (0 ou 1).
21. `NoOfImage` : Nombre d'images dans le document.
22. `NoOfCSS` : Nombre de feuilles de styles CSS liées.
23. `NoOfJS` : Nombre de scripts JavaScript externes ou inclus.
24. `NoOfSelfRef` : Nombre de liens autoréférencés.
25. `NoOfEmptyRef` : Nombre de liens vides ou ancres `#`.
26. `NoOfExternalRef` : Nombre d'hyperliens vers des domaines externes.

### 2.3. Caractéristique Catégorielle
- `TLD` : Extension de domaine extraite de l'URL, prétraitée par `TLDGrouper(top_n=20, other_label='autre')` puis encodée par `OneHotEncoder(handle_unknown='ignore')`.

### 2.4. Caractéristiques Strictement Exclues (`DERIVED_FEATURES`)
Les 6 variables dérivées suivantes sont formellement bannies de l'expérience B :
1. `URLSimilarityIndex`
2. `CharContinuationRate`
3. `TLDLegitimateProb`
4. `URLCharProb`
5. `DomainTitleMatchScore`
6. `URLTitleMatchScore`

---

## 3. Audit des Biais et Fuites de Données

1. **Biais de collecte vs fuite de cible :**
   - Les variables originales d'URL (comme `NoOfLettersInURL` et `LetterRatioInURL`) diffèrent des versions recalculées de façon systématique (0.00 % d'accord exact avec les comptages standards), révélant un algorithme propriétaire d'extraction lors de la constitution du jeu PhiUSIIL.
   - Les variables de contenu (`PAGE_FEATURES`) dépendent de la réussite du scraping au moment de la collecte : les sites de phishing hors ligne lors du crawl ont des valeurs de `LineOfCode` nulles ou faibles. Ce biais de scraping est documenté et ne constitue pas une fuite de label au sens strict, mais une corrélation structurelle propre aux données statiques.
2. **Élimination du shortcut learning :**
   - En supprimant `URLSimilarityIndex`, l'expérience B élimine le signal de similarité précalculé.
3. **Absence totale de métadonnées interdites :**
   - Les colonnes `URL`, `Host`, `Domain`, `row_id`, `FILENAME`, `phishing` et `label` sont strictement filtrées avant l'entrée dans le pipeline.

---

## 4. Protocole de Partitionnement et Validation

- **Jeu de données :** `PhiUSIIL_Phishing_URL_Dataset.csv` dédupliqué (235 370 observations).
- **Partitionnement Train / Test :**
  - **Train :** 188 492 observations (107 798 légitimes, 80 694 phishing, prévalence 42.81 %).
  - **Test :** 46 878 observations (27 052 légitimes, 19 826 phishing, prévalence 42.29 %).
  - Groupé par nom d'hôte (`Host`), garantissant une **étanchéité absolue** : 0 hôte partagé entre Train et Test.
  - Les vecteurs `row_id` sont strictement identiques à ceux des expériences A, C et D.
- **Validation croisée :** `GroupKFold(n_splits=5)` stratifié par groupe d'hôte sur Train.

---

## 5. Prétraitement et Pipelines

Chaque modèle utilise un pipeline scikit-learn standardisé :
- **Transformations numériques :**
  - Transformation $\log(1+x)$ sur les variables fortement asymétriques à queue lourde (`URLLength`, `LineOfCode`, etc.).
  - `StandardScaler()` appliqué uniquement aux modèles sensibles à l'échelle (`LogisticRegression`, `KNN`, `LinearSVC`, `SVC_RBF`).
  - Aucun scaling pour les modèles à base d'arbres (`DecisionTree`, `RandomForest`).
- **Transformation catégorielle TLD :**
  - `TLDGrouper(top_n=20, other_label='autre')` appris exclusivement sur Train.
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Intégrité :** Aucun `TargetEncoder` pour prévenir tout risque de fuite de cible.

---

## 6. Modèles Candidats et Métrique de Sélection

### 6.1. Modèles Évalués
1. **DummyClassifier** (stratégie `most_frequent`).
2. **LogisticRegression** (solver `saga`, régularisation $C \in \{0.1, 1.0, 10.0\}$).
3. **DecisionTree** (profondeurs $max\_depth \in \{5, 10, 20, None\}$).
4. **RandomForest** ($n\_estimators \in \{100, 200\}$, $max\_depth \in \{None, 20\}$).
5. **K-Nearest Neighbors (KNN)** ($n\_neighbors \in \{5, 15\}$, sous-échantillon $\le 40\,000$ lignes).
6. **LinearSVC** ($C=1.0$, `max_iter=2000`, entraîné sur tout le Train, AP calculée via `decision_function`).
7. **SVC RBF** (noyau RBF, probabilités calibrées, sous-échantillon $\le 30\,000$ lignes).

### 6.2. Calcul de l'Average Precision pour LinearSVC
`LinearSVC` n'implémente pas `predict_proba()`. Toutefois :
- `LinearSVC` fournit `decision_function(X)`, qui retourne la distance signée continue $w^T x + b$ à l'hyperplan.
- La métrique `average_precision_score(y_true, y_score)` de scikit-learn accepte directement ces scores continus de `decision_function` pour évaluer l'Average Precision (PR-AUC).
- Dans `src/train.py`, la validation croisée de `LinearSVC` utilise désormais `scoring=["average_precision", "roc_auc", "f1"]`, permettant l'évaluation homogène de l'Average Precision en CV.

### 6.3. Règle de Sélection du Meilleur Modèle
- Le meilleur modèle de B est sélectionné **exclusivement** selon la moyenne de l'**Average Precision (AP)** en validation croisée `GroupKFold` sur Train.
- Aucun résultat sur l'ensemble de Test n'est consulté pour la sélection.
- Le seuil de décision par défaut pour l'évaluation sur Test est figé à **0.5** (`standard`) pour les modèles probabilistes et **0.0** pour `LinearSVC`.

---

## 7. Définition et Évaluation des Sous-Groupes G1 et G2

Les sous-groupes sont définis rigoureusement via `get_subgroup_masks()` dans `src/evaluate.py` :

### 7.1. Définition Mathématique
- **Sous-groupe G1 :** URLs commençant exactement par le préfixe `https://www.`.
  $$\text{G1} = \{ u \in \text{Test} \mid u \text{ commence par } \texttt{"https://www."} \}$$
- **Sous-groupe G2 :** URLs de G1 ne comportant aucun chemin non vide, aucune chaîne de requête et aucun fragment (un slash final isolé `/` est autorisé).
  $$\text{G2} = \{ u \in \text{G1} \mid \text{path}(u) \in \{\text{""}, \text{"/"}\} \land \text{query}(u) = \emptyset \land \text{fragment}(u) = \emptyset \}$$

### 7.2. Propriété d'Inclusion et Décomposition en Ensembles Disjoints
Par construction :
$$\text{G2} \subseteq \text{G1}$$
Les sous-groupes ne forment **pas** une partition disjointe. L'ensemble de Test (46 878 observations) se décompose en **trois ensembles mutuellement disjoints** :
1. **$G_2$** : 27 252 observations (27 052 légitimes, 200 phishing)
2. **$G_1 \setminus G_2$** : 269 observations (0 légitime, 269 phishing)
3. **$\text{Test} \setminus G_1$** : 19 357 observations (0 légitime, 19 357 phishing)

Total vérifié :
- Légitimes Test : $27\,052 + 0 + 0 = 27\,052$
- Phishing Test : $200 + 269 + 19\,357 = 19\,826$
- Total Test : $27\,252 + 269 + 19\,357 = 46\,878$

---

## 8. Chemins des Artefacts de l'Expérience B

L'ensemble des fichiers générés lors de l'entraînement et de l'évaluation de B respecte la convention de nommage stricte :
- **Modèles :**
  - `models/B_RandomForest.joblib`
  - `models/B_LogisticRegression.joblib`
  - `models/B_DecisionTree.joblib`
  - `models/B_KNN.joblib`
  - `models/B_LinearSVC.joblib`
  - `models/B_SVC_RBF.joblib`
  - `models/best_model_B.joblib`
  - `models/model_metadata_B.json`
- **Journaux d'exécution :**
  - `reports/train_B.log`
  - `reports/evaluate_B.log`
- **Tables de résultats :**
  - `reports/tables/cv_results_B.csv`
  - `reports/tables/test_metrics_B.csv`
  - `reports/tables/subgroups_evaluation_B.csv`
  - `reports/tables/best_model_metrics_B.csv`

---

## 9. Condition d'Entraînement

L'exécution de l'entraînement de l'expérience B requiert formellement un nouveau message explicite contenant le mot-clé : **AUTORISÉ**.
En l'absence de cette autorisation explicite, aucune phase d'entraînement ne peut débuter.
