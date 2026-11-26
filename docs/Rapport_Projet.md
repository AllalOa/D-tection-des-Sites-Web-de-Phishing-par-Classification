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

## 9. Explicabilité Locale des Prédictions (SHAP / TreeSHAP)

*Volet conçu, développé et validé par : **Mohamed Makhloufi (IASD15, @mohamedmakhloufi2004)***

La conformité réglementaire (notamment l'AI Act européen pour les systèmes d'IA à haut risque) et l'acceptabilité opérationnelle par les analystes SOC (Security Operations Center) exigent qu'un modèle de détection ne se comporte pas comme une boîte noire opaque. Chaque alerte de phishing doit pouvoir être justifiée de façon transparente, interprétable et mathématiquement rigoureuse.

### 9.1 Cadre Théorique & Attribution Locale (TreeSHAP)

Les valeurs de Shapley (Shapley, 1953), issues de la théorie des jeux coopératifs à $n$ joueurs, ont été adaptées à l'apprentissage automatique par Lundberg & Lee (2017). Elles constituent l'unique méthode d'attribution locale vérifiant simultanément quatre axiomes fondamentaux :
1. **Efficacité (Additivité locale) :** La somme des attributions locales $\phi_j(x)$ de tous les descripteurs $j=1, \dots, M$ compense exactement la différence entre la prédiction du modèle $f(x)$ et la valeur de base attendue $\mathbb{E}[f(X)]$ :
   $$\sum_{j=1}^M \phi_j(x) = f(x) - \mathbb{E}[f(X)] \quad \Longleftrightarrow \quad f(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
   où $\phi_0 = \mathbb{E}[f(X)] \approx 0.4279$ sur le modèle champion de l'Expérience C (forêt aléatoire à 100 arbres).
2. **Symétrie :** Deux descripteurs apportant la même contribution marginale à toute coalition de variables reçoivent la même valeur de Shapley.
3. **Joueur Nul (Dummy Player) :** Une caractéristique n'ayant aucun impact sur la prédiction reçoit une attribution strictement nulle ($\phi_j = 0$).
4. **Consistance (Monotonie) :** Si un changement de modèle augmente la contribution marginale d'une caractéristique sans la réduire pour aucune coalition, sa valeur de Shapley ne diminue pas.

Pour les modèles arborescents (Random Forest, XGBoost), le calcul naïf des valeurs de Shapley présente une complexité combinatoire exponentielle $\mathcal{O}(M 2^{|F|})$. Nous avons exploité l'algorithme polynomial **TreeSHAP** (Lundberg et al., 2020), qui optimise le parcours récursif des sous-arbres pour calculer les attributions exactes en temps $\mathcal{O}(T L D^2)$, où $T$ est le nombre d'arbres, $L$ le nombre maximum de feuilles et $D$ la profondeur maximale.

### 9.2 Implémentation du Module `src/explain.py` & Intégration

Le module `src/explain.py` a été développé autour de la classe centrale `PhishExplainer` :
- **Extraction des 18 descripteurs lexicaux :** Extraction déterministe via `compute_url_features(url)` alignée avec le protocole `URL_ONLY_FEATURES` de l'Expérience C (excluant tout signal externe).
- **Prétraitement intégré :** Application exacte des transformations log1p du pipeline scikit-learn (`_log1p_transform`) avec remplacement systématique des valeurs manquantes (`None`, `NaN`) par `0.0`.
- **Cache d'instance TreeExplainer :** L'explainer `shap.TreeExplainer(classifier)` est instancié une seule fois par modèle et mis en cache, garantissant une latence d'inférence explicable de **~118 ms** (largement inférieure au SLA de 300 ms).
- **Vérification d'additivité à l'exécution :** Chaque calcul valide dynamiquement l'erreur résiduelle $|f(x) - (\phi_0 + \sum \phi_i)| < 10^{-4}$ (dans nos tests réels, l'erreur mesurée est de l'ordre de $5 \times 10^{-15}$).
- **Restitution multimodale :** Génération simultanée d'un dictionnaire JSON structuré (top facteurs de risque $\phi_i > 0$, top facteurs rassurants $\phi_i < 0$) et d'un graphique Waterfall matplotlib encodé en base64 pour affichage web sans accès disque.

### 9.3 Études de Cas & Interprétation des Graphiques Waterfall

```
Cas A : URL d'usurpation bancaire (Phishing avéré)
URL : http://paypal-security-update.com/login.php?user=123&token=abc&session=xyz
Prédiction : Phishing (f(x) = 0.9820, E[f(x)] = 0.4279)
Attributions TreeSHAP principales :
  + SpacialCharRatioInURL (+0.2312) -> Densité anormale de délimiteurs (&, =, ?)
  + NoOfSubDomain (+0.1840)         -> Présence de sous-domaines trompeurs
  + URLLength (+0.1250)             -> Longueur excessive masquant la cible
  + IsHTTPS = 0 (+0.0890)           -> Absence de chiffrement TLS
```

```
Cas B : URL complexe d'un service légitime
URL : https://www.google.com/search?q=machine+learning+security
Prédiction : Légitime (f(x) = 0.0410, E[f(x)] = 0.4279)
Attributions TreeSHAP principales :
  - IsHTTPS = 1 (-0.1850)           -> Protocole sécurisé valide
  - LetterRatioInURL (-0.1420)      -> Ratio élevé de caractères alphabétiques
  - ObfuscationRatio = 0 (-0.0820)  -> Absence totale de masquage
```

Cette restitution permet à un opérateur SOC de comprendre immédiatement les raisons d'un blocage et de requalifier en quelques secondes les faux positifs éventuels.

---

## 10. Limites Opérationnelles & Menaces Adversariales

*Volet conçu, rédigé et formalisé par : **Mohamed Makhloufi (IASD15, @mohamedmakhloufi2004)***

Bien que le modèle champion (Expérience C) affiche une précision et un rappel remarquables ($F_1 = 0.9936$), tout système de détection fondé sur l'apprentissage automatique présente des vulnérabilités intrinsèques face à des adversaires adaptatifs.

### 10.1 Risque d'Attaque Adversariale par Manipulation Lexicale

L'attaquant disposant d'un accès en boîte noire ou en boîte blanche au classifieur peut concevoir des URL malveillantes spécifiquement optimisées pour contourner les seuils de décision :
1. **Évasion par mimétisme lexical :** Raccourcissement délibéré du chemin d'URL, réduction du nombre de paramètres d'URL (`NoOfEqualsInURL`, `NoOfQMarkInURL`), utilisation de noms de domaine courts sans tirets.
2. **Homoglyphes et typo-squatting :** Utilisation de caractères IDN (Internationalized Domain Names) visuellement indistinguables de domaines légitimes (ex. substitution du 'o' latin par un 'о' cyrillique), ce qui altère peu les métriques lexicales brutes tout en trompant l'utilisateur final.
3. **Exploitation de services tiers légitimes :** Hébergement de pages de phishing sur des plateformes d'hébergement réputées (Google Docs, SharePoint, GitHub Pages, Firebase). Le domaine de premier niveau étant irréprochable et certifié HTTPS (`IsHTTPS = 1`), le classifieur purement lexical peut sous-estimer la menace sans analyse contextuelle du contenu.
4. **Saturation HTTPS généralisée :** L'avènement des autorités de certification automatisées et gratuites (Let's Encrypt) a neutralisé l'efficacité discriminante historique du descripteur `IsHTTPS`, les cybercriminels déployant massivement TLS sur leurs infrastructures éphémères.

### 10.2 Biais des Données Statiques & Drift Temporel

1. **Absence de dimension temporelle dans le corpus PhiUSIIL :** Le jeu de données ne comporte pas d'horodatages d'enregistrement des domaines, d'expiration des certificats SSL ni d'historique DNS. Or, la majorité des domaines de phishing présentent une durée de vie médiane inférieure à 24 heures (campagnes « burn-and-dump »).
2. **Concept Drift :** Les techniques d'obfuscation et les conventions de nommage des kits de phishing évoluent continuellement. Un modèle entraîné sur un instantané statique subira inévitablement une dégradation progressive de ses performances si un mécanisme de réentraînement incrémental continu n'est pas déployé.

### 10.3 Recommandations Stratégiques de Défense en Profondeur

Pour pallier ces limites intrinsèques, l'architecture PhishGuard doit être articulée au sein d'une approche de défense en profondeur à 3 niveaux :
- **Niveau 1 (Filtrage ultra-rapide lexical & réputationnel) :** Utilisation conjointe du modèle Expérience C et de listes blanches fiables (Tranco Top 10K, Alexa) pour filtrer 90 % du trafic à coût zéro en moins de 2 ms.
- **Niveau 2 (Validation contextuelle locale SHAP) :** Lorsqu'une URL se situe dans la zone d'incertitude décisionnelle ($0.35 \le p \le 0.65$), inspection automatique des attributions TreeSHAP pour identifier les descripteurs polarisants.
- **Niveau 3 (Enrichissement dynamique asynchrone) :** Déclenchement sélectif d'un enrichissement réseau (WHOIS domain age, réputation IP, analyse de certificat SSL) uniquement sur les requêtes ambiguës, préservant la bande passante et la rapidité du service.

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
1. Le volet d'explicabilité locale TreeSHAP a été entièrement intégré et validé par le membre IASD15 (@mohamedmakhloufi2004), conférant au système une transparence décisionnelle totale en temps réel.
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
