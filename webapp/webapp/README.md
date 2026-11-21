# PhishGuard — application web (PROJ-IASD01)

Tableau de bord + outil de test de modèle pour le projet de détection de phishing.
Elle **lit les artefacts du pipeline** (`reports/tables/*.csv`, `reports/figures/*.png`, `models/*`) ;
elle ne ré-entraîne rien.

## Lancement

```bash
pip install -r webapp/requirements.txt
python webapp/app.py                       # http://127.0.0.1:5000
# ou
flask --app webapp/app run --debug
```

Variables d'environnement (optionnelles) : `PHISHGUARD_SECRET_KEY`, `PHISHGUARD_DEBUG=1`,
`PHISHGUARD_HOST`, `PHISHGUARD_PORT`. Le mode debug est **désactivé par défaut**.
Aucune ressource externe : Bootstrap, Chart.js et les polices sont dans `static/vendor/` (fonctionne hors ligne).

## Pages et API

| Route | Description |
|---|---|
| `/` | Tableau de bord (KPI, comparaison, répartition des classes) |
| `/experiments` | Comparaison A / B / B′ / C / D + validation croisée |
| `/evaluation/<exp>` | Évaluation détaillée (`A`, `B`, `B_prime`, `C`, `D`) |
| `/limites` | Biais du jeu de données (préfixe `https://www.`, G1/G2) et limites |
| `/predict` | Test d'une URL (expériences **C** et **D** uniquement) |
| `POST /api/predict` | `{"url", "experiment"?, "threshold"?}` → classe, score, seuil, variables, avertissements |
| `GET /api/experiments` | Métriques + origine des chiffres (`csv` ou `documentation`) |
| `GET /health` | État des artefacts (modèles, tables, figures, `src.predict`) |

Codes d'erreur de l'API : `400` entrée invalide · `422` expérience A/B/B′ (variables de page requises) ·
`503` modèle ou `src/predict.py` introuvable · `500` erreur interne (message générique, trace dans les logs).

## D'où viennent les chiffres ?

`webapp/services.py` lit `reports/tables/test_metrics_<exp>.csv` (ligne RandomForest), **recalcule** précision / rappel / F1
depuis la matrice de confusion et vérifie qu'elle correspond au jeu de test du protocole
(19 826 phishing + 27 052 légitimes). Si le CSV est absent ou incohérent, les valeurs de la documentation (v1.0 §6)
servent de repli et un bandeau l'indique sur chaque page.

## Tests

```bash
pytest tests/test_webapp.py -v      # 53 tests, sans modèle ni données (prédicteur simulé)
```

## Changements par rapport à la v1

- **Métriques lues depuis `reports/tables`** (plus de valeurs codées en dur), avec contrôle de cohérence et source affichée.
- **Chiffres du jeu de données corrigés** : 235 370 URLs dédoublonnées (42,71 % / 57,29 %), cohérent avec train + test ; découpage « par hôte ».
- **Prédiction fiabilisée** : modèle par défaut = C ; A/B/B′ refusés (422) au lieu de produire une prédiction fausse ;
  seuil `standard` / `high_precision` (D) ; avertissements G1/G2 issus de l'audit des biais.
- **Nouvelle page « Biais & limites »** ; le tableau de bord n'affiche plus F1 = 99,36 % sans la référence triviale (98,80 %).
- **Robustesse** : états vides explicites, sérialisation JSON des types numpy/NaN, erreurs sans fuite d'information,
  pages d'erreur 404/405/413/500, validation stricte de l'entrée, `/health`.
- **Sécurité** : clé secrète non codée en dur, debug désactivé par défaut, en-têtes CSP / nosniff / X-Frame-Options,
  graphiques alimentés par `tojson` (pas d'injection), DOM construit avec `textContent`, route `/figures` limitée aux images.
- **Autonomie** : plus de CDN (assets locaux) ; route `/eda_figures` dupliquée supprimée.
- **Qualité** : bug du spinner (visible dès le chargement) corrigé ; accessibilité (labels, `aria-*`, lien d'évitement, tableaux avec `scope`) ;
  mobile (menu, `btn-group` qui passe à la ligne) ; axes tronqués signalés ; tests de l'application ajoutés.
