# LIMITATIONS STRUCTURELLES DU DATASET PhiUSIIL

Ce document consigne de manière strictement factuelle les artefacts structurels et limites méthodologiques identifiés lors des audits de données du projet PROJ-IASD01. Tous les chiffres ci-dessous sont issus des calculs reproductibles réalisés sur les 235 370 observations du dataset dédupliqué.

---

## 1. Structure artificielle de la classe Légitime (Noms d'hôte nus en `https://www.`)

- **Taille de la classe légitime :** 134 850 URLs.
- **Protocole et préfixe :**
  - URLs commençant par `https://www.`: 134 850 (100.00 %)
  - URLs commençant par `http://`, `http://www.`, `https://` (sans www) : 0 (0.00 %)
- **Composants d'URL :**
  - Chemin non vide (`path != '' and path != '/'`) : 0 (0.00 %)
  - URL se terminant par `/` : 0 (0.00 %)
  - Présence de paramètres de requête (`?`) : 0 (0.00 %)
  - Présence de fragment (`#`) : 0 (0.00 %)
  - Présence de port explicite : 0 (0.00 %)
  - Présence de majuscules : 0 (0.00 %)
  - Présence de caractère `@` : 0 (0.00 %)
  - Présence d'adresse IP dans l'hôte : 0 (0.00 %)
- **Comparaison avec la classe Phishing (100 520 URLs) :**
  - Chemin non vide : 27 284 URLs (27.14 %)
  - Se terminant par `/` : 36 524 URLs (36.34 %)
  - Paramètres de requête (`?`) : 6 134 URLs (6.10 %)
  - Fragment (`#`) : 869 URLs (0.86 %)
  - Adresse IP dans l'hôte : 606 URLs (0.60 %)
  - Caractère `@` : 1 506 URLs (1.50 %)
- **Performance de la règle triviale de préfixe (`URL != https://www.*` $\to$ Phishing) :**
  - Dataset complet (235 370 obs.) : Accuracy = 98.96 %, Précision = 100.00 %, Rappel = 97.57 %, FP = 0, FN = 2 445.
  - Jeu de test (46 632 obs.) : Accuracy = 99.02 %, Précision = 100.00 %, Rappel = 97.69 %, FP = 0, FN = 455.

---

## 2. Décalage d'un caractère dans les compteurs originaux du CSV

- **Écart de longueur $g = \text{URLLength\_CSV} - \text{len(URL)}$ :**
  - $g = -1$ : 186 940 observations (134 850 Légitimes [100.00 % de la classe], 52 090 Phishing [51.82 % de la classe]).
  - $g = 0$ : 48 428 observations (0 Légitime, 48 428 Phishing [48.18 % de la classe]).
  - Autres valeurs : 2 observations ($g = -35$ : 1 ligne phishing ; $g = -4$ : 1 ligne phishing).
- **Accord de la variante $s[:-1]$ sur le groupe $g = -1$ :**
  - Sur les 134 850 URLs légitimes : accord de 100.00 % absolu (0 désaccord) sur `NoOfLettersInURL`, `NoOfDegitsInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `LetterRatioInURL`, `DegitRatioInURL`, `SpacialCharRatioInURL`.
  - Sur le sous-groupe phishing $g = -1$ (52 090 URLs) : `NoOfDegitsInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL` ont 0 désaccord (100.00 %) ; `NoOfLettersInURL` a 78 désaccords (99.85 %) ; `NoOfAmpersandInURL` a 854 désaccords (98.36 %) ; `NoOfOtherSpecialCharsInURL` a 879 désaccords (98.31 %).
- **Désaccords résiduels :** Présence d'entités HTML échappées (ex : `&amp;` au lieu de `&`) dans les URLs de phishing du CSV original.

---

## 3. Comptage naïf des sous-domaines (absence de liste des suffixes publics)

- La méthode d'extraction par partition sur le point (`labels - 2` ou `count('.') - 1`) traite indifféremment les domaines de second niveau sous un suffixe public multi-niveaux.
- Exemples chiffrés :
  - Pour un domaine sous ccTLD composé (ex : `bbc.co.uk`), le nombre d'étiquettes est de 3 (`bbc`, `co`, `uk`), ce qui donne `3 - 2 = 1` sous-domaine, alors que `co.uk` constitue le suffixe public (eTLD) et `bbc` le domaine enregistré (0 sous-domaine réel).
  - Pour `google.com`, le nombre d'étiquettes est de 2 (`google`, `com`), ce qui donne `2 - 2 = 0` sous-domaine.

---

## 4. Hôtes ambigus présents dans les deux classes (Collision de labels)

- **Nombre d'hôtes distincts normalisés (`extract_host`) :** 219 094.
- **Hôtes présents uniquement dans la classe Légitime :** 134 792 (61.52 %).
- **Hôtes présents uniquement dans la classe Phishing :** 84 244 (38.45 %).
- **Hôtes présents simultanément dans les deux classes :** 58 hôtes (0.0265 % des hôtes).
- **Observations concernées par ces 58 hôtes :** 159 observations (58 légitimes, 101 phishing).
- **Exemples d'hôtes partagés les plus fréquents :**
  - `cutt.ly` (service de raccourcissement d'URL) : 36 URLs (1 légitime, 35 phishing).
  - `docsend.com` (service de partage de documents) : 5 URLs (1 légitime, 4 phishing).
  - `bestdoctors.com` : 4 URLs (1 légitime, 3 phishing).
  - `indushealthplus.com` : 4 URLs (1 légitime, 3 phishing).
  - `kashmirhills.com` : 3 URLs (1 légitime, 2 phishing).
  - `poompuhar.com` : 3 URLs (1 légitime, 2 phishing).

---

## 5. Unicité stricte des hôtes de la classe Légitime

- **Nombre d'observations légitimes :** 134 850.
- **Nombre de valeurs distinctes de `Domain` en légitime :** 134 850.
- **Nombre d'hôtes distincts (`Host`) en légitime :** 134 850.
- **Ratio d'observations par hôte en légitime :** exactement 1.0000 (aucune URL légitime ne partage son nom d'hôte avec une autre URL légitime).
- **En comparaison, pour la classe Phishing :**
  - 100 520 observations pour 84 302 hôtes distincts.
  - Ratio d'observations par hôte en phishing : 1.1924 (plusieurs URLs de phishing pointent vers une même infrastructure ou un même nom d'hôte).
