"""
src/explain.py — Moteur d'explicabilité locale SHAP (TreeExplainer).
Auteur : Mohamed Makhloufi (IASD15 / @mohamedmakhloufi2004)
Projet : PROJ-IASD01 — Détection des sites de phishing par classification

Fonctionnalités :
1. Extraction des contributions locales de Shapley via shap.TreeExplainer sur RandomForest.
2. Respect de la propriété stricte d'additivité locale :
       phi_0 + sum(phi_i) == predict_proba(x)
3. Classement des facteurs de risque (poussant vers Phishing ou vers Légitime).
4. Génération de visualisations horizontales / waterfall encodées en base64 pour la Webapp.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif sans interface graphique
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from urllib.parse import urlparse

from src.config import (
    EXPERIMENT_FEATURES,
    MODELS_DIR,
    URL_ONLY_FEATURES,
    HOST_FEATURES,
    get_experiment_tag,
)
from src.host_features import host_features
from src.url_features import compute_url_features
from src.predict import InvalidURLError, _validate_url


# ── Dictionnaire d'interprétation humaine des descripteurs ─────────────────────
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "URLLength": "Longueur totale de l'URL (caractères)",
    "DomainLength": "Longueur du nom de domaine principal",
    "IsDomainIP": "Présence d'une adresse IP brute à la place d'un domaine",
    "URLSimilarityIndex": "Indice de similarité avec les marques légitimes",
    "TLDLength": "Longueur de l'extension de premier niveau (TLD)",
    "NoOfSubDomain": "Nombre de sous-domaines (technique d'usurpation)",
    "NoOfLettersInURL": "Nombre total de lettres dans l'URL",
    "LetterRatioInURL": "Proportion de lettres par rapport à la longueur totale",
    "NoOfDegitsInURL": "Nombre total de chiffres numériques dans l'URL",
    "DegitRatioInURL": "Proportion de chiffres numériques dans l'URL",
    "NoOfEqualsInURL": "Nombre de caractères '=' (paramètres d'URL)",
    "NoOfQMarkInURL": "Nombre de points d'interrogation '?' (requêtes)",
    "NoOfAmpersandInURL": "Nombre d'esperluettes '&' (paramètres multiples)",
    "NoOfOtherSpecialCharsInURL": "Nombre de caractères spéciaux inhabituels",
    "SpacialCharRatioInURL": "Ratio de caractères spéciaux dans l'URL",
    "NoOfURLRedirect": "Nombre de redirections détectées dans l'URL",
    "NoOfSelfRedirect": "Nombre de redirections internes vers la même page",
    "ObfuscationRatio": "Ratio de masquage ou d'obfuscation de caractères",
    "NoOfObfuscatedChar": "Nombre absolu de caractères obfusqués",
    "HostLength": "Longueur du nom d'hôte",
    "HostTLDLength": "Longueur du TLD de l'hôte",
    "HostNoOfSubDomain": "Nombre de sous-domaines de l'hôte",
    "HostIsIP": "Hôte constitué d'une adresse IP",
    "HostNoOfDigits": "Nombre de chiffres dans l'hôte",
    "HostDigitRatio": "Ratio de chiffres dans l'hôte",
    "HostNoOfLetters": "Nombre de lettres dans l'hôte",
    "HostLetterRatio": "Ratio de lettres dans l'hôte",
    "HostNoOfHyphens": "Nombre de tirets dans l'hôte (typosquattage)",
    "HostLongestLabel": "Longueur du segment le plus long de l'hôte",
}


class PhishExplainer:
    """
    Moteur d'attribution locale basé sur shap.TreeExplainer pour les modèles scikit-learn.
    """

    def __init__(self, experiment: str = "C", model_path: Optional[Path] = None):
        """
        Charge le pipeline scikit-learn et initialise l'arbre TreeExplainer.
        """
        self.experiment = experiment
        self.tag = get_experiment_tag(experiment)
        
        if model_path is None:
            self.model_path = MODELS_DIR / f"best_model_{self.tag}.joblib"
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable pour l'expérience {experiment} : {self.model_path}")

        # Chargement du Pipeline
        self.pipeline = joblib.load(self.model_path)
        
        # Séparation du préprocesseur et de l'estimateur d'arbres
        if hasattr(self.pipeline, "named_steps"):
            self.preprocessor = self.pipeline.named_steps.get("preprocessor")
            self.classifier = self.pipeline.named_steps.get("classifier") or self.pipeline.named_steps.get("clf")
        else:
            self.preprocessor = None
            self.classifier = self.pipeline

        if self.classifier is None:
            raise ValueError(f"Classifieur introuvable dans le pipeline {self.model_path}")

        # Liste des features attendues selon l'expérience
        if experiment in ("C", "URL_ONLY"):
            self.feature_names = URL_ONLY_FEATURES
        elif experiment in ("D", "HOST_ONLY"):
            self.feature_names = HOST_FEATURES
        else:
            self.feature_names = EXPERIMENT_FEATURES.get(self.tag, URL_ONLY_FEATURES)

        # Initialisation de TreeExplainer sur l'ensemble d'arbres
        self.explainer = shap.TreeExplainer(self.classifier)
        
        # Valeur de base E[f(x)] pour la classe 1 (Phishing)
        if hasattr(self.explainer.expected_value, "__len__"):
            self.base_value = float(self.explainer.expected_value[1])
        else:
            self.base_value = float(self.explainer.expected_value)

    def _extract_features_df(self, url: str) -> pd.DataFrame:
        """Valide l'URL et extrait les descripteurs sous forme de DataFrame ordonné."""
        _validate_url(url)
        parsed = urlparse(url)
        if parsed.scheme and parsed.scheme.lower() not in ("http", "https"):
            raise InvalidURLError(f"Schéma '{parsed.scheme}' non supporté (seuls http et https sont acceptés).")
        if self.experiment in ("D", "HOST_ONLY"):
            feats = host_features(url)
        else:
            feats = compute_url_features(url)
            
        row = {f: feats.get(f, 0.0) for f in self.feature_names}
        return pd.DataFrame([row])[self.feature_names]

    def explain(self, url: str) -> Dict[str, Any]:
        """
        Calcule les valeurs SHAP pour une URL et vérifie l'additivité locale.
        """
        t0 = time.perf_counter()
        df_raw = self._extract_features_df(url)

        # Transformation par le préprocesseur (scaling, imputation, encodage)
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(df_raw)
        else:
            X_trans = df_raw.values

        # Prédiction de probabilité de phishing (Classe 1)
        proba_phishing = float(self.pipeline.predict_proba(df_raw)[0, 1])

        # Calcul des valeurs SHAP
        raw_shap = self.explainer.shap_values(X_trans)
        
        # Normalisation selon la forme retournée par TreeExplainer
        if isinstance(raw_shap, list):
            # Liste [classe_0, classe_1]
            phi_values = raw_shap[1][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            # Matrice (1, n_features, 2)
            phi_values = raw_shap[0, :, 1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            phi_values = raw_shap[0]
        else:
            phi_values = np.array(raw_shap).ravel()

        latency_ms = (time.perf_counter() - t0) * 1000

        # Vérification mathématique de l'additivité locale : phi_0 + sum(phi_i) == proba
        shap_sum = float(np.sum(phi_values))
        reconstructed_proba = self.base_value + shap_sum
        additivity_error = abs(proba_phishing - reconstructed_proba)

        # Dictionnaire {nom_feature: contribution_shap}
        contributions: Dict[str, float] = {}
        top_phishing: List[Dict[str, Any]] = []
        top_legitimate: List[Dict[str, Any]] = []

        for name, phi in zip(self.feature_names, phi_values):
            val_phi = float(phi)
            val_raw = float(df_raw[name].iloc[0])
            contributions[name] = val_phi
            
            entry = {
                "feature": name,
                "description": FEATURE_DESCRIPTIONS.get(name, name),
                "value": val_raw,
                "shap_value": round(val_phi, 4),
            }
            if val_phi > 0.0001:
                top_phishing.append(entry)
            elif val_phi < -0.0001:
                top_legitimate.append(entry)

        # Tri par amplitude d'impact
        top_phishing.sort(key=lambda x: x["shap_value"], reverse=True)
        top_legitimate.sort(key=lambda x: x["shap_value"])  # Le plus négatif d'abord

        return {
            "success": True,
            "url": url,
            "experiment": self.experiment,
            "predicted_proba": round(proba_phishing, 4),
            "verdict": "phishing" if proba_phishing >= 0.5 else "legitime",
            "base_value": round(self.base_value, 4),
            "shap_sum": round(shap_sum, 4),
            "reconstructed_proba": round(reconstructed_proba, 4),
            "additivity_error": float(additivity_error),
            "additivity_valid": bool(additivity_error < 1e-4),
            "latency_ms": round(latency_ms, 2),
            "contributions": contributions,
            "top_phishing_factors": top_phishing,
            "top_legitimate_factors": top_legitimate,
        }

    def generate_waterfall_plot_base64(self, url: str, max_display: int = 10) -> str:
        """
        Génère un graphique à barres horizontales SHAP haute résolution encodé en base64.
        """
        exp_res = self.explain(url)
        all_factors = (
            [(f["feature"], f["shap_value"], "phishing") for f in exp_res["top_phishing_factors"]] +
            [(f["feature"], f["shap_value"], "legitime") for f in exp_res["top_legitimate_factors"]]
        )
        
        # Tri par valeur absolue de contribution
        all_factors.sort(key=lambda x: abs(x[1]), reverse=True)
        selected = all_factors[:max_display]
        selected.reverse()  # Pour avoir la plus importante en haut

        if not selected:
            selected = [(self.feature_names[0], 0.0, "neutre")]

        names = [s[0] for s in selected]
        vals = [s[1] for s in selected]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
        fig.patch.set_facecolor("#0b0f19")
        ax.set_facecolor("#111827")

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color=colors, height=0.6, edgecolor="#1f2937", linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color="#e5e7eb", fontsize=10, fontweight="bold")
        ax.set_xlabel("Contribution SHAP (valeur de Shapley phi_i)", color="#9ca3af", fontsize=10)
        ax.tick_params(colors="#9ca3af")
        ax.axvline(0, color="#4b5563", linestyle="--", linewidth=1.0)
        ax.grid(axis="x", color="#1f2937", linestyle=":")

        for bar, val in zip(bars, vals):
            offset = 0.01 if val >= 0 else -0.01
            ha = "left" if val >= 0 else "right"
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha=ha,
                color="#f9fafb",
                fontsize=9,
                fontweight="bold"
            )

        verdict_color = "#ef4444" if exp_res["verdict"] == "phishing" else "#10b981"
        ax.set_title(
            f"Explicabilité Locale TreeSHAP — {exp_res['verdict'].upper()} (p={exp_res['predicted_proba']:.2f})\n"
            f"Base E[f(x)] = {exp_res['base_value']:.2f} | Somme phi_i = {exp_res['shap_sum']:+.2f}",
            color=verdict_color,
            fontsize=11,
            pad=12,
            fontweight="bold"
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


# ── Cache singleton en mémoire ────────────────────────────────────────────────
_EXPLAINERS: Dict[str, PhishExplainer] = {}

def get_explainer(experiment: str = "C") -> PhishExplainer:
    """Retourne l'instance de PhishExplainer mise en cache pour l'expérience donnée."""
    global _EXPLAINERS
    if experiment not in _EXPLAINERS:
        _EXPLAINERS[experiment] = PhishExplainer(experiment=experiment)
    return _EXPLAINERS[experiment]

def explain_url(url: str, experiment: str = "C") -> Dict[str, Any]:
    """Point d'entrée utilitaire rapide pour expliquer une URL."""
    return get_explainer(experiment=experiment).explain(url)
