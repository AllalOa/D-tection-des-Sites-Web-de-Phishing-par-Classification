#!/usr/bin/env python
"""
scripts/demo_predict.py — Interface CLI de prédiction pour une URL.

Usage :
    python scripts/demo_predict.py "https://example.com/login"
    python scripts/demo_predict.py "https://example.com/login" --json
    python scripts/demo_predict.py "https://example.com/login" --experiment D
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ajouter la racine du dépôt au sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_url, InvalidURLError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prédit si une URL est un site de phishing ou un site légitime."
    )
    parser.add_argument("url", type=str, help="L'URL complète à analyser")
    parser.add_argument(
        "--experiment",
        type=str,
        default="C",
        choices=["C", "D"],
        help="Modèle d'expérience à utiliser (C=URL_ONLY, D=HOST_ONLY). Défaut : C.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Seuil de décision de classification. Défaut : 0.5.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie structurée au format JSON pur sur stdout.",
    )

    args = parser.parse_args()

    try:
        raw_res = predict_url(
            args.url,
            experiment=args.experiment,
            threshold=args.threshold,
        )
        is_phishing = bool(raw_res.get("predicted_class", 0) == 1)
        verdict = "PHISHING" if is_phishing else "LEGITIME"
        
        result = {
            "url": args.url,
            "verdict": verdict,
            "is_phishing": is_phishing,
            "score": float(raw_res.get("model_score", 0.0)),
            "model_score": float(raw_res.get("model_score", 0.0)),
            "threshold": float(raw_res.get("threshold_used", args.threshold)),
            "model_name": raw_res.get("model_name", "RandomForest"),
            "experiment": raw_res.get("experiment", args.experiment),
            "features": raw_res.get("features", {}),
            "warnings": raw_res.get("warnings", []),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"URL analysée    : {args.url}")
            print(f"Modèle          : Expérience {result['experiment']} ({result['model_name']})")
            print(f"Score brut      : {result['score']:.4f} (Seuil : {result['threshold']:.4f})")
            print(f"Verdict         : {verdict}")
            if result.get("warnings"):
                print(f"Avertissements  : {', '.join(result['warnings'])}")
    except InvalidURLError as e:
        if args.json:
            print(json.dumps({"error": "InvalidURLError", "message": str(e)}))
        else:
            print(f"[ERREUR] URL invalide : {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": type(e).__name__, "message": str(e)}))
        else:
            print(f"[ERREUR] Échec de la prédiction : {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
