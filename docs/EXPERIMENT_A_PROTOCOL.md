# PROTOCOLE OPÉRATOIRE SCIENTIFIQUE — EXPÉRIENCE A (FULL FEATURES)

**Projet :** PROJ-IASD01 — Détection des Sites Web de Phishing par Classification  
**Auteur :** Machine Learning Senior Engineer  
**Date :** 2026-10-29  
**Statut :** Préparation, audit et vérification statique finalisés (Phase 6a) — En attente d'autorisation formelle d'entraînement  

---

## 1. Objectif Scientifique de l'Expérience A

L'expérience A (FULL FEATURES) a pour objectif d'évaluer le **plafond de performance théorique** en classification binaire lorsque l'intégralité des signaux précollectés est exploitable : caractéristiques lexicales de l'URL, métriques de contenu HTML et code JavaScript de la page distante, structure des hyperliens internes/externes, requêtes réseau et extension de domaine (TLD).

### Différence méthodologique essentielle avec les Expériences C et D
- **Expérience A (Caractéristiques Précollectées) :** Nécessite le téléchargement préalable du contenu de la page web distante (requêtes HTTP, rendu du DOM, extraction HTML/JS). Elle correspond à un cas d'usage d'analyse approfondie asynchrone (ex: bac à sable de sécurité / sandbox, analyse forensique de liens signalés).
- **Expériences C (URL_ONLY) et D (HOST_ONLY) :** Reposent exclusivement sur l'inspection statique de la chaîne de caractères (URL complète ou nom d'hôte normalisé). Elles sont immédiatement exécutables hors ligne et à débit réseau maximal (ex: résolveurs DNS, filtres de périmètre).

> [!WARNING]
> L'expérience A ne doit en aucun cas être présentée comme un modèle déployable à partir d'une simple URL brute sans infrastructure de crawl web active.

---

## 2. Définition et Énumération des 50 Caractéristiques Originales + TLD

Le pipeline de l'expérience A reçoit en entrée **51 colonnes au total** :
- **50 caractéristiques numériques originales** du jeu de données PhiUSIIL (`EXPERIMENT_FEATURES["A"]`).
- **1 caractéristique catégorielle** : `TLD` (extension de premier niveau).

### 2.1. Les 18 Caractéristiques Lexicales d'URL (`URL_ONLY_FEATURES`)
Colonnes originales du jeu PhiUSIIL (distinctes des 18 caractéristiques recalculées de C) :
1. `URLLength` : Longueur totale de l'URL.
2. `DomainLength` : Longueur du domaine.
3. `IsDomainIP` : Indicateur d'adresse IP dans le domaine.
4. `TLDLength` : Longueur du TLD.
5. `NoOfSubDomain` : Nombre de sous-domaines selon la définition originale.
6. `NoOfObfuscatedChar` : Nombre de caractères encodés/obfusqués.
7. `ObfuscationRatio` : Ratio d'obfuscation sur l'URL.
8. `NoOfLettersInURL` : Nombre de lettres dans l'URL.
9. `LetterRatioInURL` : Ratio de lettres dans l'URL.
10. `NoOfDegitsInURL` : Nombre de chiffres dans l'URL.
11. `DegitRatioInURL` : Ratio de chiffres dans l'URL.
12. `NoOfEqualsInURL` : Nombre de caractères `=`.
13. `NoOfQMarkInURL` : Nombre de caractères `?`.
14. `NoOfAmpersandInURL` : Nombre de caractères `&`.
15. `NoOfOtherSpecialCharsInURL` : Nombre d'autres caractères spéciaux.
16. `SpacialCharRatioInURL` : Ratio de caractères spéciaux.
17. `IsHTTPS` : Protocole HTTPS présent.
18. `HasObfuscation` : Indicateur binaire d'obfuscation.

### 2.2. Les 26 Caractéristiques de Contenu de Page (`PAGE_FEATURES`)
Caractéristiques nécessitant le téléchargement et l'analyse de la page HTML/JS :
1. `LineOfCode` : Nombre de lignes de code source.
2. `LargestLineLength` : Longueur de la plus longue ligne.
3. `HasTitle` : Présence d'une balise `<title>`.
4. `Title` : Longueur ou propriété du titre HTML.
5. `DomainTitleMatchScore` : Score de correspondance entre le domaine et le titre.
6. `URLTitleMatchScore` : Score de correspondance entre l'URL et le titre.
7. `HasFavicon` : Présence d'un favicon.
8. `Robots` : Présence ou directive robots.txt.
9. `IsResponsive` : Conception réactive/mobile.
10. `NoOfURLRedirect` : Nombre de redirections URL observées.
11. `NoOfSelfRef` : Nombre d'autoréférences / ancres locales.
12. `NoOfEmptyRef` : Nombre de références vides (`href="#"`).
13. `NoOfExternalRef` : Nombre de références externes.
14. `NoOfImage` : Nombre d'images `<img>`.
15. `NoOfCSS` : Nombre de feuilles de style CSS.
16. `NoOfJS` : Nombre de scripts JavaScript.
17. `NoOfSelfRedirect` : Nombre d'autoréacheminements.
18. `NoOfEmptyRedirect` : Nombre de redirections sans cible.
19. `NoOfExternalRedirect` : Nombre de redirections vers un domaine tiers.
20. `NoOfInternalHyperlinks` : Nombre d'hyperliens internes.
21. `NoOfExternalHyperlinks` : Nombre d'hyperliens externes.
22. `InternalHyperlinksRatio` : Ratio d'hyperliens internes.
23. `ExternalHyperlinksRatio` : Ratio d'hyperliens externes.
24. `NoOfPopups` : Nombre de pop-ups ou fenêtres surgissantes.
25. `NoOfiFrame` : Nombre d'éléments `<iframe>`.
26. `HasHiddenFields` : Présence de champs de formulaire masqués.

### 2.3. Les 6 Caractéristiques de Sécurité et Dérivées (`DERIVED_FEATURES`)
1. `HasExternalFormSubmit` : Formulaire envoyant des données à un domaine externe.
2. `HasSocialNet` : Liens vers des réseaux sociaux.
3. `HasSubmitButton` : Présence d'un bouton de soumission.
4. `HasHiddenButtons` : Présence de boutons invisibles.
5. `HasPasswordField` : Présence d'un champ `<input type="password">`.
6. `Bank` : Présence de mots-clés ou identifiants bancaires.

### 2.4. Traitement Catégoriel du `TLD`
- Extension de premier niveau (TLD) extraite de l'hôte (695 valeurs distinctes dans le corpus).
- Prétraitée via `TLDGrouper(top_n=20, other_label='autre')` : apprentissage strict sur le pli Train, les TLDs rares étant regroupés en `'autre'`.
- Encodée via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` : encodage non supervisé, évitant toute fuite de cible (`TargetEncoder` non utilisé).

---

## 3. Prétraitement et Pipeline de Données

Le prétraitement est encapsulé dans un `ColumnTransformer` scikit-learn garantissant l'absence totale de fuite entre plis :
1. **Compteurs asymétriques à queue lourde (`log_features`) :** Application de `FunctionTransformer(np.log1p)` sur les variables identifiées (`URLLength`, `LineOfCode`, `NoOfSubDomain`, etc.).
2. **Mise à l'échelle (`StandardScaler`) :** Activée uniquement pour les algorithmes sensibles aux échelles (`LogisticRegression`, `KNN`, `LinearSVC`, `SVC_RBF`). Désactivée pour les modèles basés sur des arbres décisionnels (`DecisionTree`, `RandomForest`).
3. **Données sans échelle (`pass_features`) :** Transmises telles quelles (ou normalisées si le modèle le requiert).
4. **Prétraitement catégoriel (`tld_features`) :** `TLDGrouper` + `OneHotEncoder`.

---

## 4. Protocole de Validation et Découpage

- **Source de données :** `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (235 370 observations après déduplication des 425 doublons exacts d'URL).
- **Groupement par Hôte (`GROUP_COL = "Host"`) :** Aucun nom de domaine n'est partagé entre les sous-ensembles.
- **Partitions globales :**
  - **Train :** 188 492 observations (80 694 phishing = 42.81 %, 107 798 légitimes = 57.19 %).
  - **Test :** 46 878 observations (19 826 phishing = 42.29 %, 27 052 légitimes = 57.71 %).
  - Identité stricte des `row_id` avec les partitions de C et D.
- **Validation Croisée sur le Train :**
  - `GroupKFold(n_splits=5)` groupé par `Host`.
  - 0 fuite d'hôte entre les plis d'entraînement et de validation.

---

## 5. Modèles Évalués et Sélection du Meilleur Modèle

Sept modèles sont entraînés selon le protocole unifié du projet :
1. **DummyClassifier :** Baseline majoritaire.
2. **LogisticRegression :** Régularisation $C \in \{0.1, 1.0, 10.0\}$.
3. **DecisionTree :** Profondeur maximale $\in \{5, 10, 20, \text{None}\}$.
4. **RandomForest :** Arbres $\in \{100, 200\}$, profondeur $\in \{20, \text{None}\}$.
5. **KNN :** Voisins $\in \{5, 15\}$, entraîné sur sous-échantillon stratifié de 40 000 lignes (`KNN_SUBSAMPLE_SIZE = 40000`).
6. **LinearSVC :** Entraîné sur l'intégralité du Train, évalué sans probabilités (`decision_function`).
7. **SVC_RBF :** Entraîné sur sous-échantillon stratifié de 30 000 lignes (`SVC_RBF_SUBSAMPLE_SIZE = 30000`).

### Critère de Sélection Unique
- **`REFIT_METRIC = "average_precision"` (PR-AUC)** calculée en moyenne sur les 5 plis de validation croisée du Train.
- Aucun résultat du jeu de Test n'est consulté pour le choix du modèle ou de ses hyperparamètres.

---

## 6. Politique des Seuils de Décision

- **Seuils nominaux standards :**
  - Modèles probabilistes : seuil fixe **0.5**.
  - `LinearSVC` : seuil fixe **0.0** sur `decision_function`.
- **Isolation stricte des seuils :**
  - Le seuil haute précision `0.9352` est propre à `RandomForest D`. Il est **strictement interdit** de l'appliquer à l'expérience A.
  - Tout seuil haute précision pour A devra être calculé exclusivement hors-pli (OOF) sur le Train avec GroupKFold sur `Host` avant toute exposition au Test.

---

## 7. Évaluation sur les Sous-Groupes Structurels (G1 / G2)

Après entraînement autorisé et évaluation sur le Test, les modèles seront analysés sur la partition disjointe stricte :
1. **$S_1 = G_2$ :** URLs `https://www.` avec nom d'hôte nu (27 252 obs).
2. **$S_2 = G_1 \setminus G_2$ :** URLs `https://www.` avec chemin ou requête (269 obs).
3. **$S_3 = \text{Test} \setminus G_1$ :** URLs sans `https://www.` (19 357 obs).

L'additivité arithmétique de la matrice de confusion est vérifiée sur $\{S_1, S_2, S_3\}$.

---

## 8. Cadre de Comparaison Descriptive avec C et D

La comparaison transversale finale entre A (FULL FEATURES, 50 vars), C (URL_ONLY, 18 vars) et D (HOST_ONLY, 10 vars) sera **strictement descriptive** :
- **Espaces non emboîtés :** A utilise les colonnes originales de PhiUSIIL et le TLD encodé ; C et D utilisent des caractéristiques recalculées de façon déterministe.
- **Compromis mesurés :**
  - Gain de pouvoir discriminant apporté par le contenu HTML/JS (A vs C).
  - Gain apporté par les composants d'URL hors nom d'hôte (C vs D).
  - Vulnérabilité comparée face aux biais de collecte et de scraping.

---

## 9. Limites de Validité Externe et Risques Méthodologiques Documentés

1. **Biais de collecte Page (Page Scraping Bias) :** Les sites de phishing ayant une espérance de vie brève, les données de page téléchargées lors de la création du jeu peuvent être des pages d'erreur (HTTP 404/500) ou des pages d'attente (parking pages), introduisant un signal artificiel ("page courte = phishing").
2. **Dépendance aux variables DERIVED :** Les métriques comme `DomainTitleMatchScore` supposent que la page web distante a été téléchargée et rendue.
3. **Variables originales non recalculées :** Les colonnes natives PhiUSIIL ont été conservées pour permettre la comparaison avec la littérature scientifique, malgré les anomalies lexicales identifiées en Phase 2.
4. **Latence opérationnelle :** Un modèle A ne peut être interrogé qu'après une requête réseau complète, ce qui limite son emploi aux environnements d'analyse lourde asynchrone.
