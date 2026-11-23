"""
src/research/model_manager.py - Gestionnaire de modèles optimisé en mémoire (Membre 4 - Fateh).

Fournit :
- Chargement paresseux (lazy loading) des pipelines scikit-learn (.joblib) et métadonnées (.json).
- Cache mémoire thread-safe évitant le rechargement disque coûteux (32 à 55 Mo par prédiction).
- Validation de l'intégrité des métadonnées et du schéma des caractéristiques.
- Invalidation explicite du cache.
- Préservation intégrale de la compatibilité avec l'API existante de src.predict.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    get_experiment_tag,
)

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Exception de base pour le gestionnaire de modèles."""
    pass


class ModelNotFoundError(ModelManagerError, FileNotFoundError):
    """Levée lorsqu'un modèle ou son fichier de métadonnées est introuvable."""
    pass


class InvalidModelError(ModelManagerError, ValueError):
    """Levée lorsque l'objet modèle ou les métadonnées sont invalides ou corrompus."""
    pass


class SchemaMismatchError(ModelManagerError, ValueError):
    """Levée lorsque les caractéristiques fournies ne correspondent pas au schéma attendu."""
    pass


class ModelManager:
    """
    Gestionnaire centralisé des modèles avec cache en mémoire thread-safe.
    
    Garantit un chargement unique par modèle sur le cycle de vie du processus,
    réduisant la latence d'inférence de ~250 ms à < 1 ms.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def _resolve_paths(self, experiment: Optional[str] = None) -> Tuple[str, Path, Path]:
        """Résout le tag d'expérience et les chemins du modèle et des métadonnées."""
        if experiment:
            exp_tag = get_experiment_tag(experiment)
            model_path = self.models_dir / f"best_model_{exp_tag}.joblib"
            meta_path = self.models_dir / f"model_metadata_{exp_tag}.json"
        elif BEST_MODEL_PATH.exists():
            exp_tag = "default"
            model_path = BEST_MODEL_PATH
            meta_path = MODEL_METADATA_PATH
        elif (self.models_dir / "best_model_C.joblib").exists():
            exp_tag = "C"
            model_path = self.models_dir / "best_model_C.joblib"
            meta_path = self.models_dir / "model_metadata_C.json"
        else:
            exp_tag = "default"
            model_path = BEST_MODEL_PATH
            meta_path = MODEL_METADATA_PATH

        return exp_tag, model_path, meta_path

    def _validate_metadata(self, metadata: Any, meta_path: Path) -> Dict[str, Any]:
        """Vérifie que les métadonnées sont conformes au contrat attendu."""
        if not isinstance(metadata, dict):
            raise InvalidModelError(
                f"Les métadonnées dans {meta_path} doivent être un objet JSON valide (dict), "
                f"reçu : {type(metadata).__name__}"
            )
        feature_cols = metadata.get("feature_cols")
        if not feature_cols or not isinstance(feature_cols, list):
            raise InvalidModelError(
                f"Les métadonnées dans {meta_path} ne contiennent pas de liste 'feature_cols' valide."
            )
        return metadata

    def _validate_pipeline(self, pipeline: Any, model_path: Path) -> Any:
        """Vérifie que le modèle chargé dispose au minimum de la méthode predict."""
        if not hasattr(pipeline, "predict") or not callable(getattr(pipeline, "predict")):
            raise InvalidModelError(
                f"L'objet chargé depuis {model_path} n'est pas un estimateur ou pipeline valide "
                "(méthode 'predict' manquante)."
            )
        return pipeline

    def load_model_and_metadata(
        self, experiment: Optional[str] = None, force_reload: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Charge un modèle et ses métadonnées avec mise en cache transparente.

        Parameters
        ----------
        experiment : str, optional
            Identifiant d'expérience ('C', 'D', 'A', etc.).
        force_reload : bool, default=False
            Force le rechargement depuis le disque même si déjà en cache.

        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            (pipeline_sklearn, metadonnees_dict)
        """
        exp_tag, model_path, meta_path = self._resolve_paths(experiment)

        with self._lock:
            if not force_reload and exp_tag in self._cache:
                return self._cache[exp_tag]

            if not model_path.exists():
                raise ModelNotFoundError(
                    f"Modèle introuvable : {model_path}\n"
                    f"Vérifiez la présence du fichier joblib pour l'expérience '{experiment}'."
                )
            if not meta_path.exists():
                raise ModelNotFoundError(
                    f"Métadonnées introuvables : {meta_path}\n"
                    f"Vérifiez la présence du fichier JSON pour l'expérience '{experiment}'."
                )

            try:
                pipeline = joblib.load(model_path)
            except Exception as exc:
                raise InvalidModelError(f"Erreur lors du chargement du fichier joblib {model_path}: {exc}") from exc

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:
                raise InvalidModelError(f"Erreur lors de la lecture des métadonnées {meta_path}: {exc}") from exc

            self._validate_pipeline(pipeline, model_path)
            self._validate_metadata(metadata, meta_path)

            self._cache[exp_tag] = (pipeline, metadata)
            logger.info("Modèle [%s] mis en cache avec succès depuis %s", exp_tag, model_path.name)
            return self._cache[exp_tag]

    def validate_features(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
        experiment: Optional[str] = None,
        allow_missing: bool = True,
        fill_value: float = 0.0,
    ) -> pd.DataFrame:
        """
        Valide et réordonne les caractéristiques selon le schéma exact du modèle.

        Parameters
        ----------
        features : dict or pd.DataFrame
            Caractéristiques extraites pour une ou plusieurs URLs.
        experiment : str, optional
            Expérience ciblée.
        allow_missing : bool, default=True
            Si True, complète les colonnes absentes par fill_value avec avertissement.
            Si False, lève SchemaMismatchError en cas de colonnes manquantes.
        fill_value : float, default=0.0
            Valeur de remplacement pour les colonnes manquantes.

        Returns
        -------
        pd.DataFrame
            DataFrame ordonné strictement selon feature_cols.
        """
        _, metadata = self.load_model_and_metadata(experiment)
        expected_cols: List[str] = metadata["feature_cols"]

        if isinstance(features, dict):
            feat_dict = dict(features)
            missing = [c for c in expected_cols if c not in feat_dict]
            if missing and not allow_missing:
                raise SchemaMismatchError(f"Caractéristiques obligatoires manquantes : {missing}")
            for c in missing:
                feat_dict[c] = fill_value
            return pd.DataFrame([[feat_dict[c] for c in expected_cols]], columns=expected_cols)

        elif isinstance(features, pd.DataFrame):
            df = features.copy()
            missing = [c for c in expected_cols if c not in df.columns]
            if missing and not allow_missing:
                raise SchemaMismatchError(f"Colonnes obligatoires manquantes dans le DataFrame : {missing}")
            for c in missing:
                df[c] = fill_value
            return df[expected_cols]

        else:
            raise TypeError(f"Type non supporté pour les caractéristiques : {type(features).__name__}")

    def invalidate_cache(self, experiment: Optional[str] = None) -> None:
        """
        Invalide le cache mémoire pour une expérience ou pour tous les modèles.

        Parameters
        ----------
        experiment : str, optional
            Expérience à évincer du cache. Si None, vide entièrement le cache.
        """
        with self._lock:
            if experiment is None:
                cleared_count = len(self._cache)
                self._cache.clear()
                logger.info("Cache complet vidé (%d modèles purgés)", cleared_count)
            else:
                exp_tag = get_experiment_tag(experiment)
                if exp_tag in self._cache:
                    del self._cache[exp_tag]
                    logger.info("Modèle [%s] retiré du cache", exp_tag)

    def is_cached(self, experiment: Optional[str] = None) -> bool:
        """Retourne True si le modèle spécifié est actuellement présent en mémoire."""
        exp_tag, _, _ = self._resolve_paths(experiment)
        with self._lock:
            return exp_tag in self._cache

    @property
    def cached_models(self) -> List[str]:
        """Retourne la liste des tags des modèles actuellement en cache."""
        with self._lock:
            return list(self._cache.keys())


# Singleton par défaut utilisable à travers l'application
default_model_manager = ModelManager()
