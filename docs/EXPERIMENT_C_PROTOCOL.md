# PROTOCOLE OPÉRATOIRE SCIENTIFIQUE — EXPÉRIENCE C (URL_ONLY) vs EXPÉRIENCE D (HOST_ONLY)

**Projet :** PROJ-IASD01  
**Auteur :** Machine Learning Senior Engineer  
**Date :** 2026-10-29  
**Statut :** Préparation et audit validés (Phase 5a) — En attente d'autorisation formelle d'entraînement  

---

## 1. Objectifs des Expériences C et D

Dans le cadre de l'évaluation rigoureuse de la détection de phishing sur le jeu de données PhiUSIIL (235 370 URLs dédupliquées), le projet compare deux approches d'inférence strictement autonomes (aucun appel réseau au moment de la prédiction) :

1. **Expérience C (URL_ONLY) :**
   - **Objectif :** Évaluer la capacité prédictive d'un modèle exploitant l'ensemble de la chaîne d'URL brute (schéma, nom d'hôte, chemin, paramètres de requête, fragments) au travers de 18 caractéristiques lexicales recalculées de manière déterministe.
   - **Cas d'usage :** Filtre de sécurité statique en passerelle réseau ou navigateur, capable d'inspecter l'URL complète sans télécharger la page web distante.

2. **Expérience D (HOST_ONLY) :**
   - **Objectif :** Évaluer la capacité prédictive d'un modèle restreint exclusivement aux caractéristiques structurelles du nom d'hôte normalisé (FQDN débarrassé du préfixe `www.`, des identifiants, des ports, du protocole et de tout chemin/requête).
   - **Cas d'usage :** Évaluation de la résistance aux attaques de type domaine trompeur / typosquattage et mesure de l'indépendance vis-à-vis des chemins et requêtes.

> **Note méthodologique essentielle :**
> La comparaison C vs D sera purement **descriptive**. Leurs espaces de caractéristiques ne sont pas équivalents (l'expérience C dispose d'informations sur le chemin, la requête et le protocole, tandis que l'expérience D est restreinte au nom d'hôte). La comparaison vise à quantifier le gain ou la perte de performance empirique associé à l'inclusion des composants d'URL hors hôte, à population et partitionnement identiques.

---

## 2. Définition et Énumération des Variables

### 2.1. Expérience C — 18 caractéristiques URL_ONLY
Source de données : `data/processed/url_features_all.csv` (recalculées depuis la chaîne brute par `src/url_features.py`).

| N° | Variable | Définition / Portée |
|:---|:---|:---|
| 1 | `URLLength` | Longueur totale de la chaîne d'URL (`len(URL)`). |
| 2 | `DomainLength` | Longueur du nom de domaine extrait de l'URL. |
| 3 | `IsDomainIP` | Indicateur binaire (1 si l'hôte est une adresse IPv4 ou IPv6, 0 sinon). |
| 4 | `TLDLength` | Longueur de l'extension de domaine de premier niveau (TLD). |
| 5 | `NoOfSubDomain` | Nombre de sous-domaines (étiquettes d'hôte moins le domaine principal et TLD). |
| 6 | `HasObfuscation` | Indicateur de présence de caractères obfusqués / encodés (ex: `%xx`). |
| 7 | `NoOfObfuscatedChar` | Nombre absolu de caractères obfusqués / hexadécimaux. |
| 8 | `ObfuscationRatio` | Ratio de caractères obfusqués par rapport à la longueur totale de l'URL. |
| 9 | `NoOfLettersInURL` | Nombre total de lettres alphabétiques (`[a-zA-Z]`) dans l'URL. |
| 10 | `LetterRatioInURL` | Ratio de lettres alphabétiques par rapport à la longueur totale de l'URL. |
| 11 | `NoOfDegitsInURL` | Nombre total de chiffres numériques (`[0-9]`) dans l'URL. |
| 12 | `DegitRatioInURL` | Ratio de chiffres numériques par rapport à la longueur totale de l'URL. |
| 13 | `NoOfEqualsInURL` | Nombre d'occurrences du caractère `=` dans l'URL (souvent en chaîne de requête). |
| 14 | `NoOfQMarkInURL` | Nombre d'occurrences du caractère `?` dans l'URL (délimiteur de requête). |
| 15 | `NoOfAmpersandInURL` | Nombre d'occurrences du caractère `&` dans l'URL (séparateur de paramètres). |
| 16 | `NoOfOtherSpecialCharsInURL` | Nombre de caractères spéciaux autres que `=, ?, &` (tirets, barres obliques, points, etc.). |
| 17 | `SpacialCharRatioInURL` | Ratio des caractères spéciaux par rapport à la longueur totale de l'URL. |
| 18 | `IsHTTPS` | Indicateur binaire (1 si le schéma est `https://`, 0 sinon). |

### 2.2. Expérience D — 10 caractéristiques HOST_ONLY
Source de données : `data/processed/host_features_all.csv` (calculées depuis l'hôte normalisé par `src/host_features.py`).

| N° | Variable | Définition / Portée |
|:---|:---|:---|
| 1 | `HostLength` | Longueur du nom d'hôte normalisé. |
| 2 | `HostTLDLength` | Longueur du TLD extrait de l'hôte. |
| 3 | `HostNoOfSubDomain` | Nombre de sous-domaines déduits du nom d'hôte. |
| 4 | `HostIsIP` | 1 si l'hôte normalisé est une adresse IP, 0 sinon. |
| 5 | `HostNoOfDigits` | Nombre de chiffres dans le nom d'hôte. |
| 6 | `HostDigitRatio` | Ratio de chiffres dans le nom d'hôte. |
| 7 | `HostNoOfLetters` | Nombre de lettres dans le nom d'hôte. |
| 8 | `HostLetterRatio` | Ratio de lettres dans le nom d'hôte. |
| 9 | `HostNoOfHyphens` | Nombre de tirets `-` dans le nom d'hôte. |
| 10 | `HostLongestLabel` | Longueur de l'étiquette (label) la plus longue du nom d'hôte. |

*(Note : `HostNoOfOtherSpecial` a été éliminée car strictement constante égale à 0 sur l'ensemble du corpus normalisé).*

---

## 3. Population d'Étude et Partitionnement Strict par Hôte

Pour garantir une comparabilité expérimentale sans biais, les deux expériences C et D s'appuient sur :
1. **La même population dédupliquée :** 235 370 observations (`row_id` de 0 à 235 369).
2. **Le même identifiant d'alignement :** `row_id` garantissant une correspondance ligne à ligne 1:1.
3. **Le même étiquetage binaire :** `phishing = 1` (100 520 cas), `légitime = 0` (134 850 cas).
4. **Le même découpage Train/Test (GroupShuffleSplit, 80/20, seed=42) :**
   - Regroupement strict par **Host** (`GROUP_COL = "Host"`).
   - **Aucun nom d'hôte partagé** entre Train et Test (0 chevauchement d'hôte).
   - **Aucun nom de domaine enregistrable partagé** entre Train et Test (0 chevauchement de domaine).
   - **Ensemble d'entraînement :** 188 492 observations.
     - Légitimes (0) : 107 798 (57.19 %)
     - Phishing (1) : 80 694 (42.81 %)
   - **Ensemble de test :** 46 878 observations.
     - Légitimes (0) : 27 052 (57.71 %)
     - Phishing (1) : 19 826 (42.29 %)
   - Les `row_id` du jeu de test de C sont rigoureusement identiques à ceux du jeu de test de D.

---

## 4. Métriques d'Évaluation

Pour chaque modèle, les métriques suivantes sont enregistrées de façon uniforme :
- **Average Precision (PR-AUC) :** métrique primaire de classement sous déséquilibre.
- **ROC-AUC :** capacité globale de discrimination inter-classes.
- **Precision (Précision) :** $\frac{\text{TP}}{\text{TP} + \text{FP}}$.
- **Recall (Rappel) :** $\frac{\text{TP}}{\text{TP} + \text{FN}}$.
- **Score F1 :** moyenne harmonique de la précision et du rappel.
- **Matrice de confusion brute :**
  - $\text{TP}$ : Vrais Positifs (Phishing correctement détecté).
  - $\text{FP}$ : Faux Positifs (Légitime classé à tort comme Phishing — fausse alerte).
  - $\text{TN}$ : Vrais Négatifs (Légitime correctement validé).
  - $\text{FN}$ : Faux Négatifs (Phishing non détecté — risque critique).

---

## 5. Sélection du Meilleur Modèle et Validation Croisée

1. **Validation croisée sur Train :**
   - 5 plis `GroupKFold` par `Host` (`N_FOLDS = 5`).
   - Aucun hôte partagé entre plis d'entraînement et de validation au sein d'une itération.
   - Les estimateurs et prétraitements (mise à l'échelle standard `StandardScaler`, log1p) sont fittés **exclusivement** sur le sous-ensemble d'entraînement de chaque pli (aucun ajustement sur la validation, aucun ajustement sur le test).
2. **Critère de sélection unique :**
   - Le meilleur modèle de l'expérience C est désigné **exclusivement** par la plus haute moyenne d'`average_precision` en validation croisée sur le jeu de Train.
   - Le jeu de Test reste strictement hermétique jusqu'à l'étape d'évaluation finale (Phase 5c).
3. **Modèles candidats et grilles d'hyperparamètres :**
   - `LogisticRegression` : `classifier__C` $\in [0.1, 1.0, 10.0]$
   - `DecisionTree` : `classifier__max_depth` $\in [5, 10, 20, \text{None}]$
   - `RandomForest` : `classifier__n_estimators` $\in [100, 200]$, `classifier__max_depth` $\in [\text{None}, 20]$
   - `KNN` : `classifier__n_neighbors` $\in [5, 15]$ sur sous-échantillon stratifié de 40 000 observations (limitation documentée pour contrainte de mémoire/calcul).
   - `LinearSVC` : sur ensemble Train complet.
   - `SVC_RBF` : sur sous-échantillon stratifié de 30 000 observations (modèle documenté comme distinct).
   - Baselines : `DummyClassifier` (majoritaire), `DecisionTree_depth1` (si applicable), `PrefixRule`.

---

## 6. Politique de Seuil de Décision

1. **Seuil standard opérationnel :**
   - Seuil de probabilité standard fixé à **0.5** pour tous les classifieurs probabilistes (`predict_proba`).
   - Seuil de **0.0** pour les classifieurs à marge (`decision_function`, e.g. `LinearSVC`).
2. **Interdiction de transfert de seuil :**
   - Le seuil haute précision de l'expérience D ($\tau = 0.9352$, optimisé sur le Train D pour RandomForest) est propre à D et ne doit **en aucun cas** être appliqué à l'expérience C.
   - Aucun seuil haute précision n'est pré-défini pour l'expérience C. Tout seuil spécifique devra être documenté et optimisé strictement sur le jeu de Train de C lors de la phase d'évaluation.

---

## 7. Évaluation Stratifiée sur les Sous-Groupes G1 et G2

L'analyse de robustesse requiert l'évaluation des modèles sur deux sous-groupes structurels du jeu de test (46 878 obs.) :

### 7.1. Sous-groupe G1 : URLs en `https://www.`
- **Définition :** URLs dont la chaîne débute exactement par le préfixe `https://www.`.
- **Effectif Test :** 27 521 observations.
  - Légitimes : 27 052 (100.00 % de l'ensemble des légitimes de test).
  - Phishing : 469 (2.37 % des URLs de phishing de test).
- **Enjeu :** Mesure de la capacité du modèle à discriminer le phishing mimant la convention de nommage standard des sites légitimes.

### 7.2. Sous-groupe G2 : Noms d'hôte nus en `https://www.`
- **Définition :** URLs de G1 ne comportant aucun chemin non vide (hors `/` final éventuel), aucun paramètre de requête (`?`) et aucun fragment (`#`).
- **Effectif Test :** 27 252 observations ($G2 \subseteq G1$).
  - Légitimes : 27 052 (100.00 % de l'ensemble des légitimes de test).
  - Phishing : 200 (1.01 % des URLs de phishing de test).
- **Enjeu :** Test de résistance ultime à l'artefact de construction du dataset : les URLs légitimes de PhiUSIIL sont toutes des hôtes nus en `https://www.`.

---

## 8. Baseline Structurelle PrefixRule

La règle heuristique triviale `PrefixRule` est définie par :
$$\widehat{y} = \begin{cases} 0 \text{ (légitime)} & \text{si l'URL commence par } \texttt{"https://www."} \\ 1 \text{ (phishing)} & \text{sinon} \end{cases}$$

### Comportement théorique et empirique documenté :
- Sur l'ensemble du Test global (46 878 obs.) :
  - $\text{TP} = 19\,357$, $\text{FP} = 0$, $\text{TN} = 27\,052$, $\text{FN} = 469$.
  - Précision = 1.0000, Rappel = 0.9763, F1 = 0.9880, ROC-AUC = 0.9882, PR-AUC = 0.9863.
- Sur les sous-groupes G1 et G2 :
  - Puisque toutes les URLs de G1 et G2 débutent par `https://www.`, la règle prédit systématiquement 0.
  - $\text{TP} = 0$, $\text{FP} = 0$, $\text{TN} = 27\,052$, $\text{FN}_{G1} = 469$, $\text{FN}_{G2} = 200$.
  - Rappel = 0.0000, Précision = non définie (0.0).

---

## 9. Biais et Limites Connues du Dataset PhiUSIIL

L'audit approfondi (Phases 2a-2h et Phase 4) a mis en évidence plusieurs artefacts critiques :
1. **Asymétrie de formatage protocolaire :** 100 % des URLs légitimes sont préfixées par `https://www.`, alors que les URLs de phishing présentent une diversité protocolaire (`http://`, `https://` sans www, adresses IP).
2. **Absence de chemins et requêtes dans la classe légitime :** 100 % des URLs légitimes sont des racines d'hôte (`path == ''` ou `path == '/'`, 0 query, 0 fragment). En conséquence, toute présence de requête, d'égal (`=`), d'esperluette (`&`) ou de point d'interrogation (`?`) constitue un prédicteur quasi-parfait de phishing sur ce dataset.
3. **Unicité stricte des hôtes légitimes :** Les 134 850 URLs légitimes proviennent d'exactement 134 850 hôtes distincts (ratio 1.0). En revanche, les URLs de phishing proviennent de 84 302 hôtes distincts (ratio 1.192), ce qui induit une structure de graphe différente entre les classes.
4. **Décalage d'un caractère dans le CSV d'origine :** La variable `URLLength` du CSV brut présentait un décalage systématique de -1 sur 100 % des légitimes. L'expérience C utilise des variables recalculées corrigées pour éliminer ce biais d'extraction.

---

## 10. Indépendance Protocolique et Évaluation Aveugle

- **Performances connues de D :** Les résultats de l'expérience D sur le Test sont déjà consignés dans `reports/tables/test_metrics_D.csv` (RandomForest D : PR-AUC = 0.8585, ROC-AUC = 0.8473, F1 = 0.7404 à seuil 0.5 ; F1 = 0.7588 à seuil haute précision 0.9352).
- **Règle d'intégrité :** La connaissance des résultats de D **ne doit en aucun cas influencer** la constitution des grilles d'hyperparamètres, le partitionnement ou la règle de sélection de l'expérience C.
- **Règle d'arrêt :** La phase 5a se limite strictement à l'audit et à la préparation. L'entraînement effectif de C n'aura lieu qu'après réception explicite de l'instruction contenant le mot-clé **AUTORISÉ**.
