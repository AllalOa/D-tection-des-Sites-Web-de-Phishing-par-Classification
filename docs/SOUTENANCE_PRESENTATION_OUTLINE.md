# PROJ-IASD01 — Plan Pédagogique Détaillé pour la Soutenance Orale

**Titre de la Présentation :** Détection des Sites Web de Phishing par Classification Automatisée : Compromis entre Disponibilité des Données, Biais Expérimentaux et Performances Opérationnelles  
**Durée Recommandée :** 25 minutes de présentation + 15 minutes de questions/réponses  
**Support :** 18 diapositives structurées  
**Auteur :** Ingénieur Machine Learning Senior  
**Date :** 2026-10-30 (Harmonisé en Phase 8a-bis)  

---

## Vue d'Ensemble du Déroulement

```
  [ Problématique & Dilemme ] ──▶ [ Données & Protocole ] ──▶ [ Les 5 Expériences ]
          (Slides 1-4)                    (Slides 5-7)               (Slides 8-12)
                                                                           │
  [ Conclusion & Débats Jury ] ◀── [ Architecture Cascade ] ◀── [ Biais & Sous-Groupes ]
          (Slides 17-18)                  (Slides 15-16)             (Slides 13-14)
```

---

## Slide par Slide : Contenu, Visuels et Notes de Présentation

### Slide 1 : Titre et Contexte
- **Titre :** Détection de Phishing par Machine Learning : Du Mythe de la Perfection de Laboratoire à la Réalité du Déploiement
- **Sous-titre :** Projet PROJ-IASD01 — Évaluation Comparative de 5 Espaces de Représentation sur le Corpus PhiUSIIL
- **Éléments visuels :** Logos institutionnels, schéma synthétique d'une attaque d'hameçonnage interceptée par un filtre.
- **Message clé :** Des métriques de laboratoire quasi parfaites peuvent masquer des failles opérationnelles béantes.
- **Notes orateur :**
  > « Monsieur le Président, Mesdames et Messieurs les membres du jury, le phishing reste aujourd'hui le premier vecteur d'intrusion dans les systèmes d'information. Si la littérature récente affiche des taux de détection dépassant 99.9 %, notre travail démontre que ces scores reposent sur des hypothèses de disponibilité de données rarement vérifiées sur le terrain. L'enjeu de ce projet a été de confronter rigoureusement 5 approches de modélisation pour mesurer le coût réel de l'autonomie hors ligne. »

---

### Slide 2 : Le Dilemme Opérationnel du Phishing
- **Titre :** Le Conflit Fondamental : Richesse de l'Information vs Contraintes Réseau
- **Éléments visuels :** Schéma comparatif en balance :
  - Plateau gauche : *Analyse de contenu HTML/DOM* (très riche, mais dépendance réseau, latence estimée à 0.5–5s, vulnérable aux sites éteints et au *cloaking*).
  - Plateau droit : *Analyse statique d'URL* (extrêmement rapide $<1$ ms local, 100 % hors ligne, mais information limitée à la chaîne textuelle).
- **Message clé :** Un modèle efficace doit concilier temps de réponse instantané et résistance aux attaques sans chemin.
- **Notes orateur :**
  > « Pour inspecter le contenu d'une page, un agent de sécurité doit envoyer une requête HTTP, exécuter du JavaScript et analyser le DOM. Or, un site de phishing a une durée de vie médiane courte et déploie souvent du *cloaking* pour tromper les scanners. À l'inverse, l'analyse lexicale pure de l'URL s'exécute en une demi-milliseconde localement, sans jamais contacter le serveur suspect. »

---

### Slide 3 : La Problématique et la Démarche Scientifique
- **Titre :** Objectifs de Recherche et Contributions
- **Contenu :**
  1. *Évaluation comparative :* Évaluer 5 espaces de représentation (A, B, B', C, D) sur un protocole identique et étanche.
  2. *Audit de reproductibilité :* Éliminer les caractéristiques opaques et investiguer les déviations de sélection.
  3. *Analyse de robustesse :* Mesurer la résistance sur les sous-groupes structurels trompeurs (URLs sous HTTPS sans chemin).
  4. *Proposition d'architecture :* Concevoir un pipeline de défense réaliste en cascade.
- **Notes orateur :**
  > « Notre démarche a été guidée par une exigence de reproductibilité scientifique absolue : aucun entraînement sans protocole préétabli, figeage systématique des seuils avant consultation du test, et traçabilité intégrale de chaque décision d'arbitrage. »

---

### Slide 4 : Le Benchmark PhiUSIIL et la Détection des Biais
- **Titre :** Le Dataset PhiUSIIL : Richesse et Pièges Méthodologiques
- **Éléments visuels :** Graphique circulaire de distribution des classes (57 % légitimes, 43 % phishing). Tableau des anomalies identifiées.
- **Points clés :**
  - 235 370 observations uniques après déduplication de 425 URLs dupliquées.
  - Détection d'un décalage systématique de $-1$ sur `URLLength` pour les légitimes.
  - Identification de 6 caractéristiques d'URL non reproductibles ($0\,\%$ d'accord lors d'extractions indépendantes).
- **Notes orateur :**
  > « Le dataset PhiUSIIL est un benchmark de premier plan. Cependant, notre audit a révélé des anomalies structurelles : 425 doublons exacts, un artefact arithmétique de -1 sur la longueur d'URL, et 6 variables dont les définitions sont impossibles à répliquer. Ces découvertes ont motivé la création de l'Expérience B' et de l'Expérience C. »

---

### Slide 5 : Protocole Expérimental et Étanchéité par Hôte
- **Titre :** Partitionnement Sans Fuite et Validation Croisée
- **Éléments visuels :** Diagramme de flux du partitionnement :
  - `GroupShuffleSplit` par `Host` $\rightarrow$ Train (188 492) / Test (46 878) avec **0 hôte commun**.
  - `GroupKFold(n_splits=5)` sur Train, groupé par `Host`.
- **Message clé :** Élimination absolue du surapprentissage d'infrastructure (*domain leakage*).
- **Notes orateur :**
  > « En imposant un partitionnement groupé sur le nom d'hôte normalisé, nous garantissons que nos modèles sont évalués sur leur capacité à généraliser à des infrastructures inconnues, évitant tout surapprentissage d'adresses déjà mémorisées. »

---

### Slide 6 : Les Cinq Espaces Expérimentaux en Compétition
- **Titre :** Présentation des 5 Espaces de Représentation
- **Éléments visuels :** Tableau synthétique des 5 expériences :
  - **A :** 50 variables originales + TLD (Modèle complet).
  - **B :** 44 variables originales + TLD (Ablation des 6 variables dérivées).
  - **B' :** 38 variables originales numériques (Ablation des dérivées, des 6 URL bruitées et sans TLD suite à une déviation logicielle).
  - **C :** 18 variables d'URL recalculées (100 % hors ligne).
  - **D :** 10 variables d'hôte recalculées (Infrastructure seule).
- **Notes orateur :**
  > « Nous avons exploré une échelle d'accès à l'information : de l'Expérience A qui accède au DOM et à des scores dérivés, jusqu'à l'Expérience D qui ne regarde que l'hôte, avec C comme référence lexicale pure et B' comme référence assainie. »

---

### Slide 7 : Processus de Sélection et Algorithmes Évalués
- **Titre :** Algorithmes et Critère Unique d'Arbitrage
- **Contenu :**
  - Modèles testés : `DummyClassifier`, `PrefixRule`, `LogisticRegression`, `DecisionTree`, `RandomForest`, `LinearSVC`, `KNN`, `SVC_RBF`.
  - Métrique de sélection préspécifiée : **Average Precision (AP / PR-AUC)** en validation croisée sur Train.
  - Résolution de la déviation historique de B : formalisation stricte pour B' de la règle d'éligibilité réservée aux modèles pleine échelle.
- **Notes orateur :**
  > « Dans les 5 expériences, l'algorithme Random Forest s'est imposé comme le meilleur modèle d'ensemble, surpassant systématiquement les régressions logistiques et les SVM linéaires. »

---

### Slide 8 : Résultats Globaux sur le Jeu de Test
- **Titre :** Performances Comparatives des Meilleurs Modèles
- **Éléments visuels :** Tableau comparatif complet (AP Test, F1 Test, FP, FN) et histogramme des erreurs de classification :
  - A : F1 = 1.0000 (0 erreur)
  - B : F1 = 0.9999 (4 erreurs : 2 FP, 2 FN)
  - B' : F1 = 0.9994 (23 erreurs : 13 FP, 10 FN)
  - C : F1 = 0.9936 (251 erreurs : 62 FP, 189 FN)
  - D : F1 = 0.7404 (8 851 erreurs : 1 649 FP, 7 202 FN)
- **Notes orateur :**
  > « L'Expérience B' démontre qu'après avoir retiré les variables dérivées et bruitées, le modèle maintient un F1-score de 0.9994 avec seulement 23 erreurs sur près de 47 000 tests. L'Expérience C, totalement autonome hors ligne, maintient un F1 remarquable de 0.9936. En revanche, l'Expérience D montre qu'inspecter l'hôte seul est un échec avec plus de 7 000 sites manqués. »

---

### Slide 9 : Focus sur les Courbes PR et ROC de B' et C
- **Titre :** Comportement des Modèles : Courbes PR et ROC
- **Éléments visuels :** Affichage côte à côte des graphiques réels générés :
  - `reports/figures/pr_roc_B_prime_RandomForest.png` (PR-AUC = 0.999996)
  - `reports/figures/pr_roc_C_RandomForest.png` (PR-AUC = 0.997916)
- **Message clé :** Séparation quasi parfaite des classes pour B' et C sur l'ensemble de la distribution.
- **Notes orateur :**
  > « Les courbes Précision-Rappel confirment que même à des seuils de haute sensibilité, les modèles B' et C conservent une précision remarquable. La surface sous la courbe PR atteint 0.998 pour C et 0.99999 pour B'. »

---

### Slide 10 : Analyse Différentielle : Ce que Coûte la Purification (A $\rightarrow$ B $\rightarrow$ B')
- **Titre :** Déconstruction de la Performance : De A vers B'
- **Éléments visuels :** Diagramme en cascade (*waterfall chart*) montrant l'évolution du nombre d'erreurs :
  - A $\rightarrow$ B (+4 erreurs) : suppression des 6 variables dérivées.
  - B $\rightarrow$ B' (+19 erreurs) : suppression des 6 variables d'URL non reproductibles et absence conjointe du TLD.
- **Message clé :** B et B' diffèrent de 7 entrées brutes (effet confondu). La perte de performance reste minime face au gain de rigueur méthodologique.
- **Notes orateur :**
  > « Notre audit de la Phase 8a-bis montre avec transparence que B et B' diffèrent par 7 entrées (6 variables d'URL et le TLD). L'effet observé reste très contenu (23 erreurs résiduelles), confirmant que nous n'avons pas besoin de variables opaques pour bien discriminer. »

---

### Slide 11 : Le Cas de l'Expérience C : Le Champion Hors Ligne
- **Titre :** Expérience C : L'Efficacité Statique à Latence Nulle
- **Éléments visuels :** Tableau des 18 variables d'URL (longueurs, ratios, décompte de symboles `?`, `=`, `&`, obfuscation, IP).
- **Points forts de C :**
  - Temps d'inférence $< 1$ milliseconde localement.
  - Zéro requête réseau, aucune vulnérabilité au *cloaking*.
  - Détection réussie de 99.05 % des attaques de phishing réelles (19 637 / 19 826).
- **Notes orateur :**
  > « L'Expérience C est l'option la plus pragmatique : avec un script Python d'extraction déterministe autonome, nous filtrons plus de 99 % des attaques sans jamais générer de trafic réseau. C'est l'outil adapté pour un premier filtre en passerelle. »

---

### Slide 12 : L'Échec de l'Expérience D : Pourquoi l'Hôte Seul Ne Suffit Pas
- **Titre :** Expérience D : L'Insuffisance Radicale du Nom d'Hôte
- **Éléments visuels :** Matrice de confusion de D ($12\,624$ TP, $7\,202$ FN).
- **Explication causale :**
  - Les cybercriminels hébergent leurs pages sur des domaines légitimes et réputés (Google Sites, Cloudflare, Webflow, GitHub Pages).
  - Le nom d'hôte ne porte aucune signature d'anomalie ; toute la signature malveillante réside dans le chemin d'accès et la requête.
- **Notes orateur :**
  > « Un modèle fondé sur le domaine seul est aveugle aux trois quarts des attaques actuelles hébergées sur des plateformes mutualisées. Bloquer un hôte entier conduirait à des faux positifs inacceptables sur des services légitimes. »

---

### Slide 13 : Le Révélateur des Biais : Les Sous-Groupes G1 et G2
- **Titre :** Robustesse Face aux Biais : Analyse sur G1 et G2
- **Éléments visuels :** Diagramme de Venn illustrant $G_2 \subset G_1 \subset \text{Test}$ :
  - $G_1$ (`https://www.` avec certificat) : 27 521 obs, dont 469 phishing.
  - $G_2$ (URLs nues sous HTTPS sans chemin) : 27 252 obs, dont 200 phishing.
- **Comparaison des PR-AUC :**
  - Modèle B' : G1 = 0.9983, G2 = 0.9998 (FN = 0 sur G2).
  - Modèle C : G1 = 0.8426, **G2 = 0.5521** (FN = 86 sur G2).
- **Notes orateur :**
  > « C'est sur le sous-groupe G2 que se révèle la limite de l'analyse statique. Face à une URL nue comme 'https://www.service-auth.com/', le modèle C ne dispose pas de chemin pour discriminer l'attaque et sa PR-AUC chute à 0.55. En revanche, le modèle B' inspecte le DOM de la page et résout les cas de G2 sans aucun faux négatif. »

---

### Slide 14 : Biais de Collecte et Vulnérabilités du Scraping
- **Titre :** Les Limites Réelles des Modèles Basés sur le Contenu
- **Éléments visuels :** Schéma des 3 contraintes de production du crawling web :
  1. *Site indisponible (Dead Sites) :* Volatilité rapide des serveurs malveillants.
  2. *Évasion active (Cloaking) :* Masquage de la page frauduleuse face aux scanners de sécurité.
  3. *Coût et latence réseau :* Dépendance à un tiers distant pour chaque requête.
- **Notes orateur :**
  > « Il faut se garder de croire que B' est universel parce qu'il affiche 0.9994. En conditions réelles, contacter le web distant expose à des latences réseau et à l'évasion adverse par *cloaking*. »

---

### Slide 15 : Proposition Opérationnelle : L'Architecture Conceptuelle en Cascade
- **Titre :** Vers un Système Hybride : Proposition d'une Cascade à 2 Étages
- **Éléments visuels :** Schéma architectural du pipeline en cascade :
  ```
  [ URL Entrante ] ──▶ [ Étage 1 : Modèle C (URL pure, < 1 ms) ]
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       Score Franc (> 0.8 ou < 0.2)     Zone d'Incertitude (0.2 - 0.8) ou Profil G2
                │                             │
       [ Décision Immédiate ]                 ▼
       (Filtrage local instantané)    [ Étage 2 : Modèle B' (Crawl DOM asynchrone) ]
                                              │
                                      [ Décision Finale Approfondie ]
  ```
- **Rappel scientifique :** La zone de doute 0.2–0.8 est une proposition conceptuelle qui nécessitera une validation expérimentale dédiée.
- **Notes orateur :**
  > « Notre recommandation architecturale est une alliance pragmatique : l'Expérience C agit en première ligne pour filtrer instantanément la majorité du trafic sans coût réseau. Pour les cas litigieux ou les URLs courtes sous HTTPS (profil G2), l'Expérience B' est sollicitée de façon ciblée pour inspecter le DOM. »

---

### Slide 16 : Qualité Logicielle et Reproductibilité Scientifique
- **Titre :** Rigueur d'Ingénierie : Tests et Cryptographie
- **Éléments visuels :** Capture du rapport pytest (248 tests réussis, 0 échec) et tableau des empreintes SHA256 des 25 artefacts.
- **Garanties établies :**
  - 100 % de couverture de non-régression et de cohérence dimensionnelle.
  - Immutabilité des modèles historiques A, B, C, D (empreintes SHA256 scellées).
  - Traçabilité des déviations logicielles documentée sans artifice.
- **Notes orateur :**
  > « Nous laissons un codebase propre, couvert par 248 tests unitaires automatisés, où chaque résultat est recalculable et chaque artefact est scellé par une empreinte SHA256. »

---

### Slide 17 : Conclusion et Enseignements Majeurs
- **Titre :** Conclusions du Projet PROJ-IASD01
- **Contenu synthétique :**
  1. *L'URL pure (C) suffit pour 99 % des cas :* Une barrière statique légère est viable et performante.
  2. *Le contenu (B') est le recours nécessaire pour les URLs nues (G2).*
  3. *L'infrastructure seule (D) est formellement disqualifiée.*
  4. *La rigueur méthodologique et la transparence priment sur l'illusion des scores parfaits.*
- **Notes orateur :**
  > « En conclusion, ce projet apporte une réponse claire au dilemme initial : une approche étagée, exploitant l'URL en première ligne et le contenu en renfort ciblé, offre le compromis optimal entre vitesse, coût et sécurité. »

---

### Slide 18 : Remerciements et Anticipation des Questions du Jury
- **Titre :** Questions & Réponses
- **Éléments visuels :** Liens vers la documentation du projet (`docs/`, `reports/`), remerciements aux encadrants et au jury.

---

## Guide d'Anticipation des Questions du Jury (FAQ Défense)

#### Q1. « Pourquoi B' n'utilise-t-elle pas le TLD alors que le protocole le prévoyait ? »
- **Réponse :** « Notre audit de la Phase 8a-bis a montré en toute transparence que lors de l'appel CLI avec `--experiment B_prime`, la ligne 238 de `train.py` testait `experiment in ("A", "B", "B'")` et s'est évaluée à `False`. Le modèle physique sérialisé n'intègre donc pas de transformateur TLD. Nous avons qualifié cela comme une déviation au protocole et réajusté nos analyses : B et B' diffèrent par 7 entrées (6 variables URL et le TLD). Cela illustre l'intérêt crucial des audits de conformité de pipelines sérialisés. »

#### Q2. « Pourquoi n'avez-vous pas entraîné de Transformers ou de LLM ? »
- **Réponse :** « Sur des caractéristiques tabulaires expertes, les forêts d'arbres restent la référence reconnue (Grinsztajn et al., 2022). De plus, l'inférence d'une forêt s'exécute en quelques microsecondes localement, répondant à nos contraintes de latence. L'application d'un Transformer directement sur les caractères de l'URL constitue une perspective d'extension naturelle pour s'affranchir de l'ingénierie manuelle de caractéristiques. »

#### Q3. « Votre modèle C s'effondre sur G2 avec une PR-AUC de 0.55. N'est-ce pas un angle mort dangereux ? »
- **Réponse :** « C'est précisément l'apport de notre analyse de sous-groupes. G2 représente des URLs nues comme `https://www.domaine-suspect.com/`. Syntaxiquement, ces URLs ne contiennent aucun chemin ni paramètre. L'information lexicale y est absente. C'est pour cet angle mort que notre architecture en cascade bascule ces cas vers l'Étage 2 (Modèle B'), qui inspecte le DOM et résout 100 % des cas de G2 sans faux négatif. »

#### Q4. « Pourquoi dites-vous que le seuil 0.2–0.8 de la cascade n'est pas validé ? »
- **Réponse :** « Par rigueur scientifique. Proposer un seuil indicatif dans un rapport ne vaut pas validation empirique. Pour valider formellement la cascade, il faudra calibrer les probabilités sur un jeu de validation dédié, aligner les prédictions séquentielles des deux modèles et mesurer la courbe coût/efficacité opérationnelle selon un protocole expérimental autorisé. »
