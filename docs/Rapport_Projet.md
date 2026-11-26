# Rapport Scientifique & Technique — PROJ-IASD01
## Détection des Sites Web de Phishing par Classification Supervisée

**Formation :** Master IASD / SIAD  
**Unité d'Enseignement :** Data Mining & MLOps  
**Date :** Novembre 2026  
**Jeu de Données :** PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, id 967, CC BY 4.0)  
**Auteurs :** Équipe PROJ-IASD01 (Groupe SIAD / IASD)  

---

## Sommaire

1. [Introduction & Contexte de Cybersécurité](#1-introduction--contexte-de-cybersécurité)
2. [Description & Structure du Jeu de Données PhiUSIIL](#2-description--structure-du-jeu-de-données-phiusiil)
3. [Contrôle Qualité & Assainissement des Données](#3-contrôle-qualité--assainissement-des-données)
4. [Analyse Exploratoire des Données (EDA)](#4-analyse-exploratoire-des-données-eda)
5. [Validation Statistique Formelle des Hypothèses](#5-validation-statistique-formelle-des-hypothèses)
6. [Méthodologie de Modélisation & Étanchéité Anti-Fuite](#6-méthodologie-de-modélisation--étanchéité-anti-fuite)
7. [Protocoles Expérimentaux (Expériences A, B, B′, C, D)](#7-protocoles-expérimentaux-expériences-a-b-b-c-d)
8. [Résultats Expérimentaux & Benchmark Comparatif](#8-résultats-expérimentaux--benchmark-comparatif)
9. [Explicabilité Locale des Prédictions (SHAP / LIME)](#9-explicabilité-locale-des-prédictions-shap--lime)
10. [Limites Opérationnelles & Menaces Adversariales](#10-limites-opérationnelles--menaces-adversariales)
11. [Architecture Logicielle & Déploiement Applicatif](#11-architecture-logicielle--déploiement-applicatif)
12. [Conclusion & Perspectives](#12-conclusion--perspectives)
13. [Bibliographie & Références Scientifiques](#13-bibliographie--références-scientifiques)

---

## 1. Introduction & Contexte de Cybersécurité

### 1.1 Problématique de la Menace Phishing
Le phishing demeure le vecteur d'attaque initial prédominant dans plus de 80 % des compromissions de sécurité d'entreprise selon les rapports récents de l'ENISA et du Verizon DBIR. Les cybercriminels déploient des répliques trompeuses d'institutions bancaires, d'interfaces de messagerie ou de passerelles d'authentification dans l'objectif d'extorquer des identifiants d'accès ou des données de paiement.

Traditionnellement, la protection périmétrique repose sur des **listes noires d'URLs** (Google Safe Browsing, PhishTank, OpenPhish). Si ces mécanismes garantissent un taux de faux positifs quasi nul, leur délai médian d'indexation (souvent supérieur à 24 voire 48 heures) s'avère rédhibitoire face à des campagnes d'attaque éphémères dont l'espérance de vie médiane ne dépasse pas 4 à 8 heures.

### 1.2 Objectif du Projet
L'objectif central de **PROJ-IASD01** est de concevoir, valider statistiquement et déployer un système de classification supervisée capable d'analyser en temps réel une URL candidate et de formuler un verdict binaire immédiat (**Légitime** vs **Phishing**) avec une très haute précision ($F_1 > 0.99$), tout en isolant strictement les descripteurs lexicaux autonomes de toute dépendance au téléchargement réseau de la page web cible.

---

## 2. Description & Structure du Jeu de Données PhiUSIIL

Le corpus **PhiUSIIL** (*Phishing URL Information Security Dataset*), publié au dépôt UCI Machine Learning sous l'identifiant 967 par Prasad et Chandra (2024), rassemble **235 795 observations** et **56 colonnes**.

Chaque enregistrement correspond à une URL collectée entre 2020 et 2022 depuis des sources de référence vérifiées (DMOZ, Alexa Top 1M pour les sites légitimes ; PhishTank, OpenPhish pour les sites malveillants).

### Taxonomie des 56 Variables
Les colonnes se répartissent en quatre sous-ensembles fonctionnels :
1. **Descripteurs d'URL stricts (`URL_ONLY` - 18 variables) :** Longueur totale (`URLLength`), longueur du domaine (`DomainLength`), indicateur d'adresse IP (`IsDomainIP`), longueur de l'extension (`TLDLength`), nombre de sous-domaines (`NoOfSubDomain`), caractères obfusqués (`NoOfObfuscatedChar`), ratios de lettres/chiffres/caractères spéciaux, protocole (`IsHTTPS`).
2. **Descripteurs de Page Web (`PAGE` - 32 variables) :** Volume du code source (`LineOfCode`), présence d'iframe (`NoOfiFrame`), balises images/CSS/JS, champs mot de passe (`HasPasswordField`), formulaires sans action (`HasSubmitButton`), méta-données SEO (`HasDescription`, `HasSocialNet`).
3. **Descripteurs Dérivés (`DERIVED` - 6 variables) :** Calculs composites tels que `URLSimilarityIndex` ou `CharContinuationRate`.
4. **Métadonnées & Cible (6 colonnes) :** `FILENAME`, `URL`, `Domain`, `TLD`, `Title`, et l'étiquette binaire `label` (recodée `phishing = 1 - label`).

---

## 3. Contrôle Qualité & Assainissement des Données

Conformément au standard CRISP-DM (*Cross-Industry Standard Process for Data Mining*), une phase rigoureuse de contrôle d'intégrité a été conduite :

1. **Déduplication Stricte :** Le dataset brut comporte 425 doublons exacts sur la chaîne d'URL. Ces lignes ont été purgées pour obtenir un ensemble de **235 370 observations uniques**.
2. **Contrôle d'Exhaustivité :** Aucune valeur manquante (`NaN` ou `null`) n'a été détectée sur l'intégralité des 56 colonnes (complétude de 100.0 %).
3. **Contrôle d'Hôtes Distincts :** 219 094 noms d'hôtes uniques composent le dataset épuré.
4. **Équilibre des Classes :** Le jeu dédoublonné présente 100 520 URLs de phishing (42.71 %) et 134 850 URLs légitimes (57.29 %), assurant un équilibre favorable exempt de déséquilibre critique nécessitant des techniques de rééchantillonnage artificiel (SMOTE).

---

## 4. Analyse Exploratoire des Données (EDA)

L'exploration approfondie des distributions a mis en lumière trois phénomènes statistiques majeurs :

### 4.1 Asymétrie Sévère des Compteurs (Heavy-Tail)
Les distributions de variables telles que `URLLength`, `LineOfCode`, `NoOfImage` ou `NoOfLettersInURL` présentent un coefficient d'asymétrie (*skewness*) largement supérieur à 5, avec des queues de distribution s'étirant sur plusieurs milliers d'unités. Une transformation non linéaire concave `log1p(x) = \ln(1 + |x|)` a été systématiquement insérée dans le pipeline scikit-learn pour stabiliser la variance sans supprimer les observations extrêmes qui constituent un signal prédictif précieux.

### 4.2 Divergence Majeure Pearson vs Spearman
La comparaison des coefficients de corrélation linéaire de Pearson ($r$) et des rangs de Spearman ($\rho$) avec la variable cible a révélé un écart spectaculaire. Par exemple, pour `NoOfExternalRef`, le coefficient de Pearson ne vaut que $-0.258$, tandis que le coefficient de Spearman atteint $+0.842$ ($p < 10^{-300}$). Ce comportement prouve formellement la nature non-linéaire et purement monotone des descripteurs, invalidant les modèles purement linéaires au profit des algorithmes d'arbres décisionnels.

### 4.3 Biais Structurel du Protocole `https://www.`
L'EDA a identifié un artefact expérimental intrinsèque au corpus PhiUSIIL : **100 % des sites légitimes** débutent par `https://www.`, alors que **seuls 2.4 % des sites de phishing** comportent cette séquence complète (46.2 % utilisent `https://` sans sous-domaine `www`, et 51.4 % recourent au protocole HTTP non chiffré). Un classifieur naïf s'appuyant uniquement sur cette règle obtient trivialement un score $F_1$ de 0.9880. Nos travaux ont donc spécifiquement isolé cette composante dans l'expérience D pour garantir une véritable robustesse sur les sous-domaines complexes.

---

## 5. Validation Statistique Formelle des Hypothèses

Pour satisfaire aux plus hautes exigences de rigueur scientifique, 56 tests d'hypothèses formels ont été calculés et consignés dans `reports/tables/phase4_hypothesis_tests.csv`.

### 5.1 Test de Normalité de Shapiro-Wilk
- **Hypothèse $H_0$ :** La distribution de la variable suit une loi normale $\mathcal{N}(\mu, \sigma^2)$.
- **Hypothèse $H_1$ :** La distribution dévie significativement de la normalité.
- **Résultats :** Sur les variables clés (`URLLength`, `DomainLength`, `LineOfCode`), la statistique $W$ s'établit entre 0.42 et 0.81 ($p < 10^{-10}$).
- **Interprétation :** Rejet catégorique de l'hypothèse de normalité à $\alpha = 0.01$. Ce résultat justifie impérieusement l'abandon des tests paramétriques de Student au profit de tests de rangs non paramétriques.

### 5.2 Test Non-Paramétrique de Mann-Whitney $U$
- **Hypothèse $H_0$ :** Les distributions du descripteur $X$ pour les classes Phishing et Légitime sont stochastiquement identiques.
- **Hypothèse $H_1$ :** Les distributions diffèrent significativement.
- **Résultats :** Testé sur 31 variables continues. Pour `URLSimilarityIndex`, $U = 1.35 \times 10^{10}$, $p < 10^{-300}$, avec une corrélation bisériale de rang $|r| = 0.9922$ (taille d'effet extrême). Pour `URLLength`, $|r| = 0.061$, confirmant un pouvoir séparateur plus modeste sur la longueur brute seule.

### 5.3 Test d'Indépendance du Chi-Deux ($\chi^2$)
- **Hypothèse $H_0$ :** La caractéristique binaire est statistiquement indépendante de la classe Phishing.
- **Hypothèse $H_1$ :** Il existe une association significative entre la caractéristique et la classe.
- **Résultats :** Pour `HasSocialNet`, $\chi^2 = 1.45 \times 10^5$ ($dof=1$, $p < 10^{-300}$), avec un $V$ de Cramér de 0.7839 (association forte). Pour `IsDomainIP`, $\chi^2 = 853.2$ ($p < 10^{-180}$), $V = 0.0602$.

### 5.4 Tests Multi-Groupes : Kruskal-Wallis & ANOVA
- **Kruskal-Wallis $H$ sur les extensions (Top 5 TLDs : .com, .org, .net, .app, .uk) :** $H = 214.5$, $p < 10^{-40}$, taille d'effet $\eta^2 = 0.14$. Rejet de l'homogénéité de longueur d'URL selon le bureau d'enregistrement.
- **ANOVA à 1 facteur :** $F = 38.7$, $p < 10^{-25}$, confirmant une divergence significative des moyennes inter-TLD.
- **Contrôle des Fausses Découvertes :** Toutes les p-values ont fait l'objet de corrections par la méthode de Bonferroni et par la procédure FDR de Benjamini-Hochberg.

---

## 6. Méthodologie de Modélisation & Étanchéité Anti-Fuite

### 6.1 Découpage par Hôte (Zero Data Leakage)
L'erreur méthodologique classique dans l'analyse d'URLs réside dans le découpage aléatoire standard (`train_test_split`), qui distribue des URLs appartenant au même domaine à la fois dans le train et dans le test. Le modèle apprend alors par cœur la réputation du domaine plutôt que des règles générales de classification.

Pour éliminer ce risque :
- Le partitionnement utilise `GroupShuffleSplit(test_size=0.20, random_state=42)` groupé par nom d'hôte (`Host`).
- Le jeu d'entraînement compte **188 492 URLs** et le jeu de test scellé **46 878 URLs**.
- L'intersection entre les hôtes du train et du test est rigoureusement égale à **zéro**, attesté mathématiquement par la suite de tests `tests/test_data_leakage.py`.

### 6.2 Pipeline scikit-learn Hermétique
L'ensemble des étapes de transformation est scellé dans un `Pipeline` unique composé d'un `ColumnTransformer` :
1. Branche Log-Transform : `FunctionTransformer(np.log1p)` sur les variables compteurs.
2. Branche Normalisation : `StandardScaler` conditionnel, activé exclusivement pour les algorithmes sensibles à l'échelle (Régression Logistique, KNN, SVM).
3. Branche Catégorielle : `TLDGrouper(top_n=20)` regroupant les extensions rares en `autre`, suivi d'un `OneHotEncoder(handle_unknown='ignore')`.
4. Évaluateur final : Classifieur supervisé (Random Forest ou Boosting).

---

## 7. Protocoles Expérimentaux (Expériences A, B, B′, C, D)

Cinq protocoles distincts ont été modélisés pour cerner le compromis entre performance théorique et applicabilité opérationnelle :
- **Expérience A (Full Features) :** 50 variables numériques + TLD. Référence théorique absolue exploitant le code HTML téléchargé.
- **Expérience B (Sans DERIVED) :** 44 variables + TLD. Retrait des 6 indicateurs dérivés pour tester la robustesse face à l'absence de scores pré-calculés.
- **Expérience B′ (Purified Content) :** 38 variables (sans TLD). Suppression des variables d'URL instables pour tester la dépendance exclusive au contenu de page.
- **Expérience C (URL_ONLY — Champion Déployable) :** 18 variables calculées à partir de la seule chaîne d'URL brute. **100 % autonome hors réseau, inférence en moins d'une milliseconde**.
- **Expérience D (Host_ONLY) :** 8 descripteurs dérivés exclusivement du nom de domaine/hôte.

---

## 8. Résultats Expérimentaux & Benchmark Comparatif

Les modèles ont été entraînés sous validation croisée 5 plis (`GroupKFold` par hôte) et évalués une seule fois sur le jeu de test indépendant scellé (46 878 URLs : 19 826 Phishing, 27 052 Légitimes).

### Synthèse des Performances

| Protocole | Modèle Optimal | $F_1$-Score | PR-AUC | ROC-AUC | Précision | Rappel | Faux Positifs | Faux Négatifs | Latence |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp. A** | RandomForest (100) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 450 ms (réseau) |
| **Exp. B** | RandomForest (200) | **0.9999** | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 2 | 2 | 450 ms (réseau) |
| **Exp. B′** | RandomForest (max_d=20) | **0.9994** | 0.9999 | 0.9999 | 0.9993 | 0.9995 | 13 | 10 | 450 ms (réseau) |
| **Exp. C** | **RandomForest (100)** | **0.9936** | **0.9979** | **0.9974** | **0.9969** | **0.9905** | **62** | **189** | **1.4 ms (offline)** |
| **Exp. C (XGBoost)** | XGBoost (opt) | **0.9944** | 0.9982 | 0.9978 | 0.9965 | 0.9923 | 69 | 153 | **0.8 ms (offline)** |
| **Exp. D** | RandomForest (HP) | **0.7408** | 0.8585 | 0.8473 | 0.8845 | 0.6368 | 1 649 | 7 202 | 0.5 ms (offline) |

L'Expérience C constitue le **champion opérationnel** : elle n'accuse qu'une perte marginale de 0.6 % de $F_1$ par rapport aux modèles lourds nécessitant le scraping web, tout en divisant la latence globale par plus de 300 et en éliminant les risques de sécurité inhérents au téléchargement de pages malveillantes.

---

## 9. Explicabilité Locale des Prédictions (SHAP / LIME)

> [!NOTE]
> **SECTION RÉSERVÉE — MEMBRE IASD15 (@mohamedmakhloufi2004)**  
> Conformément à la répartition des contributions formalisée dans `MEMBER_TASKS.md`, la rédaction approfondie de cette section, l'implémentation du module `src/explain.py` et les visualisations Waterfall SHAP sont dévolues à **mohamedmakhloufi2004** pour valoriser sa contribution technique à hauteur de 100 %.

### 9.1 Cadre Théorique & Attribution Locale (TreeSHAP)
Les valeurs de Shapley issues de la théorie des jeux coopératifs satisfont des propriétés axiomatiques uniques : efficacité, symétrie, joueur nul et additivité :
$$\sum_{j=1}^M \phi_j(x) = f(x) - \mathbb{E}[f(X)]$$
Pour les ensembles d'arbres, l'algorithme polynomial **TreeSHAP** (Lundberg et al., 2020) permet d'extraire la contribution signée de chaque descripteur lexical à la décision d'alerte.

### 9.2 Études de Cas & Graphiques Waterfall (En cours par IASD15)
- **Cas 1 : URL d'usurpation de marque (Phishing avéré) :** Décomposition de la contribution de `NoOfSubDomain`, `SpacialCharRatioInURL` et de la présence de tokens suspects (`login`, `paypal`).
- **Cas 2 : Faux Positif potentiel sur domaine complexe :** Analyse des facteurs ayant poussé le modèle vers une prédiction erronée.

---

## 10. Limites Opérationnelles & Menaces Adversariales

> [!NOTE]
> **SECTION RÉSERVÉE — MEMBRE IASD15 (@mohamedmakhloufi2004)**  
> Cette section analytique est confiée au membre **mohamedmakhloufi2004** (IASD15). Elle sera enrichie par ses soins pour détailler les vulnérabilités de robustesse et les attaques par évasion.

### 10.1 Risque d'Attaque Adversariale par Manipulation Lexicale
Un attaquant averti peut concevoir une URL de phishing imitant les descripteurs des domaines légitimes :
- Allongement artificiel du domaine pour tromper les ratios d'entropie.
- Emploi de certificats Let's Encrypt valides pour saturer `IsHTTPS = 1`.
- Suppression des tirets et des caractères spéciaux dans le chemin d'accès.

### 10.2 Biais des Données Statiques
Le corpus PhiUSIIL ne comporte pas de signal temporel (horodatage de création WHOIS, date d'enregistrement du certificat SSL). L'intégration de ces flux dynamiques constitue une recommandation prioritaire pour un passage en production industrielle.

---

## 11. Architecture Logicielle & Déploiement Applicatif

L'application **PhishGuard** a été architecturée selon les principes de séparation des préoccupations :
- **Couche Données & Services (`webapp/services.py`) :** Chargement asynchrone, validation de cohérence avec les rapports générés, traitement par lot tolérant aux erreurs.
- **Interface Utilisateur Flask (`webapp/app.py` & templates Jinja2) :** Thème sombre natif, navigation latérale avec les 4 volets académiques, visualisations dynamiques Chart.js et images haute résolution auto-hébergées (aucune dépendance CDN externe).
- **Inférence par Lot :** Formulaire de téléversement CSV avec traitement paginé et export des prédictions labellisées en CSV.
- **Suite de Tests Dédiée :** 57 tests unitaires Flask garantissant la résilience des endpoints HTTP et de l'API REST JSON.

---

## 12. Conclusion & Perspectives

Le projet **PROJ-IASD01** a démontré la faisabilité et la redoutable efficacité d'une détection de phishing purement lexicale et autonome. En atteignant un score $F_1$ de **0.9936** et une PR-AUC de **0.9979** sans aucun accès réseau, le modèle issu de l'Expérience C offre une solution de pointe pour les passerelles de messagerie et les extensions de navigateur.

Les axes futurs d'évolution porteront sur :
1. La finalisation du volet d'explicabilité locale SHAP par le membre IASD15.
2. L'enrichissement avec des réseaux de neurones récurrents (Char-CNN / Bi-LSTM) pour capturer les représentations sous-motiques sans ingénierie manuelle de caractéristiques.
3. Le déploiement d'une API conteneurisée Docker avec cache Redis pour atteindre une cadence supérieure à 10 000 requêtes par seconde.

---

## 13. Bibliographie & Références Scientifiques

1. **Prasad, A., & Chandra, S. (2024).** *PhiUSIIL: A novel dataset for phishing URL detection using machine learning and deep learning techniques.* Computers & Security, Elsevier. UCI Machine Learning Repository, id=967.
2. **Lundberg, S. M., et al. (2020).** *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56-67.
3. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* In ACM SIGKDD (pp. 1135-1144).
4. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
5. **Breiman, L. (2001).** *Random Forests.* Machine Learning, 45(1), 5-32.
6. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* In ACM SIGKDD (pp. 785-794).
