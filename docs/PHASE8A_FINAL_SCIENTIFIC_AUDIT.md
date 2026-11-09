# PROJ-IASD01 — Phase 8a : Rapport de Consolidation Scientifique Finale des Cinq Expériences

**Date :** 2026-10-30 (Harmonisé en Phase 8a-bis)  
**Statut :** Validé — Audit consolidé des résultats, protocoles, biais et reproductibilité  
**Auteur :** Ingénieur Machine Learning Senior, Rédacteur Scientifique  
**Projet :** Détection des Sites Web de Phishing par Classification — Dataset PhiUSIIL  

---

## 1. Contexte Général et Principes d'Audit

Le projet **PROJ-IASD01** visait à évaluer la détectabilité des attaques de phishing à travers différents niveaux d'abstraction et d'accès aux données. Cinq espaces de représentation ont été formalisés et entraînés sous un protocole expérimental rigoureux :
- **A (FULL_FEATURES) :** 50 caractéristiques numériques brutes + TLD (URL, Contenu HTML, Réseau et Dérivées).
- **B (URL_PAGE) :** 44 caractéristiques numériques brutes + TLD (exclusion des 6 variables dérivées suspectes).
- **B' (PURIFIED_CONTENT) :** 38 caractéristiques numériques brutes sans TLD (exclusion des variables dérivées, des 6 variables textuelles d'URL non reproductibles, et exclusion technique du TLD documentée en Phase 8a-bis).
- **C (URL_ONLY) :** 18 caractéristiques d'URL recalculées de manière déterministe et autonome.
- **D (HOST_ONLY) :** 10 caractéristiques d'infrastructure extraites exclusivement du nom d'hôte normalisé.

Le présent rapport consolide l'ensemble des résultats de ces cinq expériences, vérifie l'intégrité arithmétique et cryptographique des artefacts, analyse les biais expérimentaux inhérents au dataset PhiUSIIL, et évalue de manière critique l'architecture de déploiement en cascade.

---

## 2. Audit Mathématique et Alignement des Partitions

### 2.1. Partitionnement et Étanchéité
Toutes les expériences partagent rigoureusement le même partitionnement issu de `src/split.py` :
- **Dataset complet dédupliqué :** 235 370 observations uniques (425 doublons exacts éliminés).
- **Jeu d'entraînement (Train) :** 188 492 observations (80 694 phishing, 107 798 légitimes).
- **Jeu de test (Test) :** 46 878 observations (19 826 phishing, 27 052 légitimes).
- **Stratification par groupe :** Regroupement par `Host` (`GroupShuffleSplit` à 20 % pour Test, `GroupKFold(n_splits=5)` pour Train).
- **Étanchéité inter-hôtes :** Exactement **0 hôte commun** entre Train et Test, et 0 hôte commun entre plis d'entraînement.
- **Identité des observations :** Les vecteurs d'identifiants `row_id` et de labels cibles $y$ sont strictement identiques entre A, B, B', C et D.

### 2.2. Contrôle Arithmétique des Matrices de Confusion Test
Pour chaque meilleur modèle (`RandomForest`), les métriques enregistrées dans `reports/tables/test_metrics_*.csv` ont été vérifiées à partir des nombres entiers purs $\text{TP}, \text{FP}, \text{TN}, \text{FN}$ :

$$\text{Précision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Rappel} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1-score} = \frac{2 \cdot \text{Précision} \cdot \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

| Exp. | TP | FP | TN | FN | Total | Phishing Réels ($\text{TP}+\text{FN}$) | Légitimes Réels ($\text{TN}+\text{FP}$) | Précision Calculée | Rappel Calculé | F1-Score Calculé | Concordance Fichier |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | 19 826 | 0 | 27 052 | 0 | 46 878 | 19 826 | 27 052 | 1.00000000 | 1.00000000 | 1.00000000 | **Exacte ($10^{-16}$)** |
| **B** | 19 824 | 2 | 27 050 | 2 | 46 878 | 19 826 | 27 052 | 0.99989912 | 0.99989912 | 0.99989912 | **Exacte ($10^{-16}$)** |
| **B'** | 19 816 | 13 | 27 039 | 10 | 46 878 | 19 826 | 27 052 | 0.99934439 | 0.99949561 | 0.99942000 | **Exacte ($10^{-16}$)** |
| **C** | 19 637 | 62 | 26 990 | 189 | 46 878 | 19 826 | 27 052 | 0.99685263 | 0.99046706 | 0.99364959 | **Exacte ($10^{-16}$)** |
| **D** | 12 624 | 1 649 | 25 403 | 7 202 | 46 878 | 19 826 | 27 052 | 0.88446718 | 0.63673963 | 0.74043227 | **Exacte ($10^{-16}$)** |

Toutes les identités scalaires sont parfaitement exactes. Aucune divergence ni incohérence arithmétique n'est présente.

---

## 3. Analyse Comparative et Différentielle des Représentations

### 3.1. Expérience A vs Expérience B : L'Impact des Variables Dérivées
- **Modification :** Retrait des 6 `DERIVED_FEATURES` (`URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `DomainTitleMatchScore`, `URLTitleMatchScore`).
- **Constat :**
  - Dans A, le modèle atteint la perfection formelle sur Test ($\text{F1} = 1.0000$, $\text{FP}=0$, $\text{FN}=0$).
  - Dans B, la suppression de ces 6 variables ne fait apparaître que **4 erreurs au total** ($\text{FP}=2$, $\text{FN}=2$, $\text{F1} = 0.9999$).
- **Interprétation scientifique :** Les variables dérivées contenaient une forte redondance descriptive. Leur retrait démontre empiriquement que l'ensemble combiné {URL + Contenu de la Page} sépare presque parfaitement les classes au sein de ce corpus sans nécessiter ces attributs dérivés.

### 3.2. Expérience B vs Expérience B' : Nettoyage d'URL et Absence Conjointe de TLD
> **Mise au point méthodologique issue de la Phase 8a-bis :**  
> B et B' diffèrent par **7 entrées brutes** (6 variables numériques d'URL exclues et 1 variable catégorielle TLD absente du modèle B'). La comparaison B/B' ne constitue donc **pas une ablation univariée contrôlée** des seules caractéristiques d'URL.
- **Constat :**
  - Le nombre d'erreurs sur Test passe de 4 dans B à **23 dans B'** ($\text{FP}=13$, $\text{FN}=10$, $\text{F1} = 0.9994$).
  - La performance reste extrêmement élevée ($\text{AP Test} = 0.999996$).
- **Interprétation scientifique :** Les 6 caractéristiques d'URL éliminées présentaient des définitions opaques et un taux d'accord de $0\,\%$ lors de recalculs indépendants. L'effet observé (+19 erreurs) reflète l'effet combiné du retrait de ces 6 caractéristiques bruitées et de l'absence de l'encodage du TLD. Pour dissocier strictement ces deux effets, un entraînement avec TLD serait requis (non exécuté afin de respecter l'interdiction de nouveaux fits).

### 3.3. Expérience B' vs Expérience C : Représentation Mixte vs URL Statique Pure
> **Mise en garde méthodologique essentielle :**  
> La comparaison entre B' et C **ne constitue pas une ablation contrôlée pure des seules variables de page (`PAGE_FEATURES`)**, car les variables d'URL de B' (12 variables issues du CSV) et celles de C (18 variables recalculées par `src/url_features.py`) ne sont pas strictement superposables. De plus, une corrélation descriptive ne démontre pas à elle seule une relation causale directe.
- **Différence fonctionnelle :**
  - **B' (38 variables) :** Combine 12 variables syntaxiques d'URL et 26 variables de structure HTML du DOM. Exige en pratique d'avoir accès au contenu de la page.
  - **C (18 variables) :** Exploite exclusivement la chaîne textuelle de l'URL. Fonctionne à 100 % hors ligne avec un temps de calcul sub-milliseconde.
- **Différence empirique :**
  - C produit 251 erreurs sur Test ($\text{FP}=62$, $\text{FN}=189$, $\text{F1} = 0.9936$), contre 23 erreurs pour B' ($\text{F1} = 0.9994$).
  - La différence de 228 détections supplémentaires dans B' reflète l'apport d'information du DOM dans ce jeu de données, au prix de la dépendance à la disponibilité du site distant.

### 3.4. Expérience C vs Expérience D : URL Complète vs Infrastructure du Nom d'Hôte
- **Modification :** Réduction de l'espace au seul nom d'hôte normalisé (10 caractéristiques dans D contre 18 dans C).
- **Constat :**
  - Chute massive des performances dans D : l'Average Precision Test s'effondre à $0.8585$ et le Rappel à $0.6367$ ($\text{FN} = 7\,202$).
  - Près de **36.3 % des sites de phishing ne sont pas détectés** par le modèle D seul.
- **Interprétation scientifique :** Le nom d'hôte seul est insuffisant pour caractériser une attaque de phishing. Une proportion majeure d'attaques modernes est hébergée sur des plateformes d'hébergement ou des services cloud mutualisés légitimes où l'hôte est anodin et où la charge utile réside dans le chemin et les paramètres de l'URL.

---

## 4. Audit des Biais Expérimentaux et Limites de Généralisation

L'évaluation des modèles dans un benchmark académique tel que PhiUSIIL présente des biais structurels majeurs :

### 4.1. Le Biais du Préfixe `https://www.`
Dans le dataset PhiUSIIL, une forte asymétrie de distribution prévaut :
- La totalité des sites légitimes du jeu de test (27 052 sur 27 052, soit $100\,\%$) démarrent par le préfixe `https://www.`.
- À l'inverse, seuls 469 sites de phishing sur 19 826 ($2.36\,\%$) adoptent ce préfixe dans le jeu de test.
- Une règle naïve monocorrespondance (`PrefixRule`) atteint à elle seule un F1-score Test de **0.9880** sans aucun apprentissage statistique. Cette corrélation massive crée un raccourci d'apprentissage artificiel.

### 4.2. L'Artefact de Mesure sur `URLLength`
L'analyse menée en Phase 4 a mis en évidence un biais systématique de mesure dans le fichier original :
- La variable `URLLength` du CSV présente un décalage systématique de $-1$ sur les URLs légitimes par rapport à la longueur textuelle réelle calculée en Python (`len(URL)`).
- Ce biais non documenté a conduit à l'exclusion de `URLLength` dans l'Expérience C et l'Expérience B'.

### 4.3. Dépendance Temporelle et Vulnérabilités Pratiques du Scraping (A, B, B')
Les caractéristiques de page (`LineOfCode`, `HasPasswordField`, `Bank`, `Pay`, `NoOfiFrame`, etc.) souffrent de contraintes opérationnelles majeures en production :
1. **Volatilité des sites de phishing :** La durée de vie d'un site malveillant est souvent éphémère (médiane $< 24$ h). Si un système d'analyse tente de crawler une page déjà désactivée ou suspendue, les 26 variables de contenu sont indisponibles.
2. **Techniques d'Évasion (Cloaking) :** Les kits de phishing intègrent fréquemment des protections anti-bots : si le visiteur est identifié comme un scanner d'IP connue ou un crawler headless, le serveur renvoie une page d'erreur 404 ou une page neutre.
3. **Latence opérationnelle :** Réaliser une requête HTTP distante, suivre les redirections et analyser le DOM introduit une latence de réseau (généralement comprise entre 500 ms et quelques secondes dans la littérature réseau), alors que l'extraction lexicale de l'URL s'effectue en temps quasi-nul localement.

### 4.4. Étanchéité par Hôte vs Généralisation aux Nouvelles Campagnes
Le découpage `GroupKFold` par `Host` garantit l'absence de fuite d'apprentissage (*data leakage*) au sein de la distribution de PhiUSIIL. Toutefois :
- Il ne protège pas contre la **dérive de distribution temporelle (*covariate shift*)**.
- Les attaquants adaptent continuellement leurs stratégies lexicales et leurs infrastructures.
- Des scores de validation croisée supérieurs à $0.999$ sur ce benchmark ne préjugent pas d'une performance équivalente face à des flux de trafic réels en production.

---

## 5. Analyse des Sous-Groupes Structurels G1 et G2

Les sous-groupes structurels ont été isolés pour évaluer la robustesse des modèles face à des URLs trompeuses :
- **Sous-groupe G1 :** URLs commençant par `https://www.` (27 521 observations de Test, dont 469 phishing).
- **Sous-groupe G2 :** URLs de G1 dépourvues de chemin non vide, de requête ou de fragment (27 252 observations de Test, dont 200 phishing).
- **Propriété mathématique :** $G_2 \subseteq G_1$. G1 et G2 ne sont **jamais additionnés**, car ils ne forment pas des ensembles disjoints.

| Expérience | Modèle | G1 PR-AUC [IC 95%] | G1 ROC-AUC [IC 95%] | G1 FP / FN ($s=0.5$) | G2 PR-AUC [IC 95%] | G2 ROC-AUC [IC 95%] | G2 FP / FN ($s=0.5$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | RandomForest | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | FP=0 / FN=0 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | FP=0 / FN=0 |
| **B** | RandomForest | 0.99999 [1.0000, 1.0000] | 0.99999 [1.0000, 1.0000] | FP=2 / FN=0 | 0.99995 [0.9998, 1.0000] | 0.99999 [1.0000, 1.0000] | FP=2 / FN=0 |
| **B'** | RandomForest | 0.99831 [0.9951, 0.9999] | 0.99994 [0.9998, 1.0000] | FP=13 / FN=3 | 0.99976 [0.9992, 1.0000] | 0.99999 [1.0000, 1.0000] | FP=13 / FN=0 |
| **C** | RandomForest | 0.84263 [0.8069, 0.8724] | 0.93676 [0.9187, 0.9533] | FP=62 / FN=86 | 0.55207 [0.4823, 0.6213] | 0.85175 [0.8124, 0.8861] | FP=62 / FN=86 |
| **D** | RandomForest | 0.27268 [0.2231, 0.3302] | 0.72501 [0.6873, 0.7603] | FP=1649 / FN=286 | 0.22537 [0.1624, 0.2958] | 0.76574 [0.7252, 0.8064] | FP=1649 / FN=108 |

### Impact de la Faible Prévalence du Phishing
- Dans G1, la prévalence du phishing est de **1.70 %** (469 / 27 521) ; dans G2, elle est de **0.73 %** (200 / 27 252).
- Dans ce contexte d'asymétrie forte, la PR-AUC est la métrique clé.
- Le modèle C (URL seule) subit une baisse notable sur G2 ($\text{PR-AUC} = 0.5521$), car une URL nue comme `https://www.service-login.com/` n'offre presque aucune caractéristique lexicale hors du nom de domaine. Les modèles exploitant le contenu (B, B') maintiennent une PR-AUC $> 0.999$ sur ce sous-groupe.

---

## 6. Audit des Processus de Sélection et Traçabilité des Protocoles

### 6.1. Critère de Sélection Standardisé
Dans les 5 expériences, le critère d'optimisation préspécifié a été l'**Average Precision (AP)** en validation croisée `GroupKFold` 5 plis sur Train :
$$\text{REFIT\_METRIC} = \text{"average\_precision"}$$

### 6.2. Traçabilité de la Déviation Historique dans l'Expérience B
- Lors de l'audit de la Phase 7a, il a été démontré que dans `src/train.py`, le modèle `SVC_RBF` affichait une AP CV de **0.99999979**, supérieure à celle de `RandomForest` (**0.99999902**).
- Cependant, `RandomForest` a été sélectionné comme `best_model_B.joblib` car le script `src/train.py` ne comportait aucun bloc conditionnel de mise à jour pour `SVC_RBF`.
- Cette exclusion technique des modèles sous-échantillonnés n'était pas annoncée dans le texte du protocole initial (`docs/EXPERIMENT_B_PROTOCOL.md`). Cette déviation est maintenue dans les rapports pour garantir la traçabilité historique.

### 6.3. Formalisation Préalable pour l'Expérience B'
Pour l'Expérience B', la règle d'éligibilité a été formellement actée dans [EXPERIMENT_BPRIME_PROTOCOL.md](file:///c:/Users/PC/Desktop/mini_projet_Data/PROJ-IASD01/docs/EXPERIMENT_BPRIME_PROTOCOL.md) avant tout entraînement :
- Seuls les modèles pleine échelle (`LogisticRegression`, `DecisionTree`, `RandomForest`, `LinearSVC`) étaient éligibles au titre de meilleur modèle.
- `RandomForest` a remporté la première place absolue ($\text{AP} = 0.999994$), devançant également les modèles sous-échantillonnés (`SVC_RBF` à $0.999971$).

### 6.4. Qualification de la Déviation sur le TLD dans l'Expérience B'
L'audit approfondi de la Phase 8a-bis a formellement qualifié l'écart de configuration de B' :
- Dans `src/train.py` (ligne 238), la condition d'activation du TLD était codée sous la forme `use_tld = experiment in ("A", "B", "B'")`.
- L'appel CLI ayant été exécuté avec `--experiment B_prime`, l'expression s'est évaluée à `False`.
- **Qualification :** Il s'agit d'une **déviation au protocole préétabli**, qui prévoyait 38 caractéristiques + TLD. Le modèle physique `best_model_B_prime.joblib` a été réellement entraîné sur 38 caractéristiques numériques pures (sans transformateur TLD).
- Lors de l'évaluation dans `src/evaluate.py`, le `ColumnTransformer` (configuré par défaut avec `remainder='drop'`) a ignoré la colonne TLD présente dans les données de test, prédisant exclusivement sur les 38 variables numériques d'apprentissage.

---

## 7. Évaluation Critique de l'Architecture en Cascade Proposée

Une architecture conceptuelle en cascade a été formulée pour concilier rapidité opérationnelle et profondeur d'analyse :
- **Étage 1 :** Modèle C (18 variables d'URL) $\rightarrow$ Inférence locale rapide, aucun appel réseau.
- **Étage 2 :** Modèle B' (38 variables numériques) $\rightarrow$ Inspection approfondie du DOM pour les cas litigieux.

### Réserves Méthodologiques Indispensables :
1. **Absence de validation expérimentale du seuil d'incertitude :**  
   L'intervalle de doute indicatif ($0.2 < \hat{p} < 0.8$) **n'a pas fait l'objet d'une validation expérimentale formelle**.
2. **Statut de la proposition :**  
   Cette cascade constitue une proposition architecturale prometteuse mais non démontrée empiriquement sur un banc d'essai conjoint. Il est proscrit d'affirmer des taux de filtrage sans protocole expérimental dédié.

---

## 8. Contrôle Cryptographique d'Intégrité SHA256 (25 Artefacts)

L'intégrité physique des 25 fichiers clés du projet a été contrôlée sans altération :

| Artefact Physique du Projet | Empreinte SHA256 Intégrale | Statut |
|---|---|:---:|
| `models/best_model_A.joblib` | `73f7d9612d43ebfb92b6931fb50b3abe97a0a8b69b922298eee0c1d7f85d05b7` | Conforme |
| `models/best_model_B.joblib` | `7ada1a61744f4fd5def06a5fe4a9192ccf731390a62a29103e0655dad8fff85f` | Conforme |
| `models/best_model_B_prime.joblib` | `7876db407f7777be76333ce421c011cdff5a818e34423b341d2d071a3559daf5` | Conforme |
| `models/best_model_C.joblib` | `9dfac9513af2ae835fca867a2e9e164a249543371b1687bb5cb298390fddd093` | Conforme |
| `models/best_model_D.joblib` | `560c107f9d1e0a585e0c056e70451dce132d1ff297fbeff3d6465382f193ca18` | Conforme |
| `models/model_metadata_A.json` | `e9e11e96083010b912410861ec8ef0fb838374180880e84a67df0b4189aacbaf` | Conforme |
| `models/model_metadata_B.json` | `4cd904d6a97c3e54f385a771ec0c0ee5ca541898b3b7b4afc5bd86d8fba702c7` | Conforme |
| `models/model_metadata_B_prime.json` | `624d71eb91571ffa68827fa22b0bcdb505988a4d317919041a348e9bf08797cb` | Conforme |
| `models/model_metadata_C.json` | `357f896d791fb485f0a25b2c0e8c5e6dabfc9087bd8875b7087414a1aa6bd86c` | Conforme |
| `models/model_metadata_D.json` | `050a3d84927af66ee3407be3c8ab69a7d5c89ffe34837d7118f5a63581a2917f` | Conforme |
| `reports/tables/cv_results_A.csv` | `8087ae327753271b76b56cc1d078cabd69ce827b7e35daeb2bf6f660af3dea3e` | Conforme |
| `reports/tables/cv_results_B.csv` | `6e19fa21e51b29b45f87f74f26ae58f9781e4f507138987ea5061a97f624047a` | Conforme |
| `reports/tables/cv_results_B_prime.csv` | `c6541d63bf0d92081c651519749849ec7c5543f03ae62b6d6a5b8a6cebaa6cc3` | Conforme |
| `reports/tables/cv_results_C.csv` | `252ea8282d677e45cb4656bb1b85f9a4c8ef9da69527a15af202dbcc5733edba` | Conforme |
| `reports/tables/cv_results_D.csv` | `e3703d0afd2f805590b64937225cf97232ec45861b75036a6bb6db76a6e9bbf0` | Conforme |
| `reports/tables/test_metrics_A.csv` | `2d689c7cc19cedb7b65de37036753775b9586a653f462f83d2ec03516a4bb685` | Conforme |
| `reports/tables/test_metrics_B.csv` | `7d4ea14b9315c2d6458c72aa0186b44df3fe709005be06f182613c45bf4665d2` | Conforme |
| `reports/tables/test_metrics_B_prime.csv` | `b4a9e8eb9889744c3c72ce4f51467e108b28f30688caaf13039268b2ff082e2e` | Conforme |
| `reports/tables/test_metrics_C.csv` | `504a1a78e30a8693f74991ebadee1259bc15a0635300b5d1b8398e82805b24ed` | Conforme |
| `reports/tables/test_metrics_D.csv` | `a9843fcdeb40967275ba57f217f1e1d7beac6f72d06fa47ef4a4883b1dce1f19` | Conforme |
| `reports/tables/subgroups_evaluation_A.csv` | `5957eec700ee02cbcb2ee1c60315bc51e3cba3a4a4bb40d2ca129cb04ea90fd0` | Conforme |
| `reports/tables/subgroups_evaluation_B.csv` | `b2936aa714e6a353f9da9683489d35bd1f7994cd4f61f5abadbe3561d059b657` | Conforme |
| `reports/tables/subgroups_evaluation_B_prime.csv` | `2adfa2be581823faa606d7dbe651383ad484794460882dc72d95c3c551c9625b` | Conforme |
| `reports/tables/subgroups_evaluation_C.csv` | `9e1b2114fc7a464d7a8fdc555b9bb96bc01cb4398e38b806a2188f991852a332` | Conforme |
| `reports/tables/subgroups_evaluation_D.csv` | `4850ed2a1f3e27fea38eac16c1a229335b309544c7f2d82b82a308f9f81886bd` | Conforme |

---

## 9. Conclusion

La consolidation scientifique finale de la Phase 8a et de son audit 8a-bis confirme l'intégrité et la rigueur du projet PROJ-IASD01 :
1. Les cinq modèles `best_model_*.joblib` et leurs métriques sont arithmétiquement vérifiés et cryptographiquement scellés.
2. Les déviations logicielles (B et B') ont été identifiées, expliquées par le code et documentées sans réécriture historique.
3. Les résultats et limites sont présentés avec la retenue et l'exactitude requises par l'évaluation scientifique.
