# Feuille de Route & Spécifications — Membre IASD15 (@mohamedmakhloufi2004)
**Projet :** PROJ-IASD01 — Détection des sites web de phishing par classification  
**Objectif :** Rééquilibrer l'impact de contribution de 57% (note actuelle : 8.52/20) vers **100% d'impact (note cible : 18.0 - 20.0/20)**.

Ce document décrit avec précision les 4 briques techniques dont la réalisation exclusive incombe à **mohamedmakhloufi2004 (IASD15)**.

---

## Tâche 1 : Module d'Explicabilité Locale (`src/explain.py`)

### 1.1 Contexte et Objectif
La prédiction binaire d'un modèle de classification (Phishing vs Légitime) doit être interprétable localement par un analyste en cybersécurité ou un utilisateur final. L'objectif est d'implémenter l'attribution locale des descripteurs avec la bibliothèque **SHAP** (`shap.TreeExplainer`) et, si possible, **LIME** (`lime.lime_tabular.LimeTabularExplainer`).

### 1.2 Fichiers Concernés
* `src/explain.py` (à créer)
* `requirements.txt` (s'assurer de la présence de `shap>=0.44.0` et `lime>=0.2.0.1`)

### 1.3 Spécification Technique & Interface Attendue
Le module doit fournir une classe `PhishExplainer` avec l'API suivante :

```python
"""
src/explain.py — Moteur d'explicabilité locale SHAP / LIME.
Auteur assigné : mohamedmakhloufi2004 (IASD15)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap

class PhishExplainer:
    def __init__(self, model_pipeline, background_data: Optional[pd.DataFrame] = None):
        """
        Initialise l'explainer SHAP (TreeExplainer) sur le classifieur du pipeline.
        
        Parameters
        ----------
        model_pipeline : Pipeline scikit-learn fitté (ex: models/best_model_C.joblib)
        background_data : pd.DataFrame, optionnel pour le calcul de la valeur de base E[f(x)]
        """
        ...

    def explain_url_features(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Calcule les valeurs SHAP pour un vecteur de caractéristiques d'une URL.
        
        Returns
        -------
        dict avec :
            - base_value : float (espérance de la prédiction E[f(x)])
            - prediction_proba : float (probabilité prédite de phishing)
            - shap_values : Dict[str, float] (contribution signée de chaque feature)
            - top_phishing_factors : List[Tuple[str, float, float]] (features poussant vers phishing)
            - top_legitimate_factors : List[Tuple[str, float, float]] (features poussant vers légitime)
        """
        ...

    def generate_waterfall_plot_base64(self, features_dict: Dict[str, float]) -> str:
        """
        Génère un graphique Waterfall SHAP encodé en base64 pour intégration directe dans la webapp.
        """
        ...
```

### 1.4 Critères d'Acceptation
1. Respect strict de la propriété d'additivité locale :  
   $$\sum_{i=1}^M \phi_i + \phi_0 = f(x)$$
2. Temps d'exécution de l'explication locale $\le 300\text{ ms}$ pour une URL donnée.
3. Gestion propre des types numpy / float natifs pour compatibilité JSON.

---

## Tâche 2 : Volet Explicabilité dans la Webapp (`webapp/webapp/`)

### 2.1 Contexte et Objectif
Intégrer les visualisations SHAP directement dans l'application web pour que l'utilisateur puisse inspecter visuellement les contributions lors du test d'une URL.

### 2.2 Fichiers Concernés
* `webapp/webapp/app.py` (ajouter la route `/explain` et enrichir `/api/predict`)
* `webapp/webapp/templates/predict.html` (ajouter l'accordéon ou la carte visuelle d'explicabilité SHAP)
* `webapp/webapp/static/css/style.css` (styles des badges de contribution positive/négative)

### 2.3 Spécification Fonctionnelle
1. Dans la page « Tester une URL », sous le résultat de prédiction (Score / Verdict), afficher :
   - Un graphique **Waterfall SHAP** ou un diagramme horizontal à barres des 5 variables les plus influentes.
   - Un tableau explicatif en langage clair :  
     *Exemple :* `SpacialCharRatioInURL = 0.28 (+0.24 vers Phishing)` car ce ratio est anormalement élevé par rapport aux sites légitimes.
2. Ajouter un endpoint API `/api/explain` recevant `{"url": "..."}` et retournant les contributions SHAP au format JSON.

---

## Tâche 3 : Tests Unitaires de l'Explicabilité (`tests/test_explain.py`)

### 3.1 Contexte et Objectif
Assurer une couverture de tests automatisée par `pytest` pour garantir la robustesse du moteur d'explication.

### 3.2 Fichier Concerné
* `tests/test_explain.py` (à créer)

### 3.3 Cas de Tests Obligatoires
1. `test_explainer_initialization()` : Vérifie que le TreeExplainer s'initialise correctement avec le modèle RandomForest de l'expérience C.
2. `test_shap_additivity()` : Vérifie que la somme des valeurs SHAP + valeur de base est égale à la probabilité / marge de décision à $10^{-4}$ près.
3. `test_top_factors_ordering()` : Vérifie que les facteurs de risque sont correctement triés par valeur absolue de contribution.
4. `test_waterfall_plot_generation()` : Vérifie que l'image PNG / base64 est valide et non vide.
5. `test_edge_cases()` : Vérifie la robustesse face à une URL très courte (`http://a.co`) ou avec des caractères encodés.

---

## Tâche 4 : Sections « Explicabilité » et « Limites » du Rapport

### 4.1 Contexte et Objectif
Rédiger les chapitres analytiques du rapport scientifique démontrant la maîtrise des enjeux d'interprétabilité et les vulnérabilités du système.

### 4.2 Fichier Concerné
* `docs/Rapport_Projet.md` (sections réservées à compléter)

### 4.3 Contenu Attendu
1. **Section « 8. Explicabilité Locale des Prédictions » (2 à 3 pages) :**
   - Théorie des valeurs de Shapley appliquées aux arbres de décision (TreeSHAP).
   - Analyse comparative globale des features : quelles variables poussent le plus souvent vers le phishing ?
   - Étude de 2 cas concrets détaillés :
     * Cas 1 : Vrai Positif (URL de phishing par usurpation de marque PayPal/Microsoft).
     * Cas 2 : Faux Positif ou cas limite (site légitime avec URL complexe ou sous-domaines multiples).
   - Comparaison méthodologique : avantages et inconvénients de SHAP vs LIME.
2. **Section « 9. Limites Opérationnelles et Risques Adversariaux » (2 pages) :**
   - Biais intrinsèques du dataset PhiUSIIL (absence de capture dynamique, dépendance aux variables HTML).
   - Attaques par évasion (*Adversarial URL generation*) : comment un attaquant peut manipuler la longueur, les tirets ou les sous-domaines pour tromper le classifieur.
   - Recommandations pour un déploiement réel (mise à jour continue des modèles, intégration avec des flux de menaces TI).

---

## Synthèse du Plan d'Action pour IASD15

| Tâche | Livrable principal | Commits conventionnels suggérés |
|---|---|---|
| **Tâche 1** | `src/explain.py` | `feat(explain): implement SHAP TreeExplainer for URL classification` |
| **Tâche 2** | Webapp SHAP UI | `feat(webapp): add SHAP waterfall visualization in prediction view` |
| **Tâche 3** | `tests/test_explain.py` | `test(explain): add unit tests for SHAP additivity and feature attribution` |
| **Tâche 4** | Sections 8 & 9 du rapport | `docs(report): author explainability and adversarial limitations chapters` |
