"""
host_features.py - Extraction de caractéristiques de l'hôte (Host Features - Expérience D).

Ce module extrait le nom d'hôte nettoyé et normalisé d'une URL et calcule 11 caractéristiques
strictement limitées à la structure de l'hôte, sans fuite de schéma, de chemin, ou d'artefacts
de protocole.

Fonctions exportées :
    extract_host(url: str) -> str
    host_features(url: str) -> dict
    host_features_batch(urls: pd.Series | list) -> pd.DataFrame
"""

import ipaddress
import re
from typing import Dict, List, Union
import numpy as np
import pandas as pd

from src.predict import InvalidURLError

HOST_FEATURES: List[str] = [
    "HostLength",
    "HostTLDLength",
    "HostNoOfSubDomain",
    "HostIsIP",
    "HostNoOfDigits",
    "HostDigitRatio",
    "HostNoOfLetters",
    "HostLetterRatio",
    "HostNoOfHyphens",
    "HostLongestLabel",
]


def _is_ip_address(host: str) -> bool:
    """Retourne True si host est une adresse IPv4 ou IPv6 valide."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_host(url: str) -> str:
    """
    Extrait le nom d'hôte normalisé d'une URL.

    Règles appliquées :
    - Minuscules
    - Sans schéma (http, https, ou tout schéma avec '://')
    - Sans identifiants utilisateur (ex: 'user:pass@host' -> 'host')
    - Sans port (ex: ':8080' supprimé)
    - Sans chemin, requête ni fragment (ex: '/path?q=1#hash' supprimé)
    - Sans 'www.' initial
    - Sans point final (ex: 'example.com.' -> 'example.com')
    - Tolère les entrées sans schéma (ex: 'x.com', 'www.x.com/abc') pour les tests
    - Lève InvalidURLError si l'hôte est vide ou invalide.
    """
    if url is None or not isinstance(url, str):
        raise InvalidURLError(f"L'URL doit être une chaîne non vide (reçu : {type(url).__name__})")

    s = url.strip()
    if not s:
        raise InvalidURLError("L'URL est vide ou ne contient que des espaces")

    # 1. Retirer le schéma si présent (ex: http://, https://, ftp://, etc.)
    if "://" in s:
        s = s.split("://", 1)[1]
    elif s.startswith("//"):
        s = s[2:]

    # 2. Retirer le chemin, la requête et le fragment
    s = re.split(r"[/?#]", s, maxsplit=1)[0]

    # 3. Retirer les identifiants utilisateur (avant '@')
    if "@" in s:
        s = s.rsplit("@", 1)[-1]

    # 4. Retirer le port
    # Gestion des adresses IPv6 entre crochets : [::1]:8080
    if s.startswith("[") and "]" in s:
        host = s[1:s.index("]")]
    else:
        host = s.split(":", 1)[0]

    # 5. Minuscules
    host = host.lower()

    # 6. Retirer "www." initial (répété si nécessaire ex: www.www. -> hors www initial)
    if host.startswith("www."):
        host = host[4:]

    # 7. Retirer le point final éventuel
    host = host.rstrip(".")

    # 8. Validation finale : hôte non vide
    if not host:
        raise InvalidURLError("L'hôte extrait est vide ou invalide")

    return host


def host_features(url: str) -> Dict[str, Union[int, float]]:
    """
    Calcule exactement les 11 caractéristiques d'hôte pour une URL.

    Variables calculées :
    - HostLength : longueur de l'hôte
    - HostTLDLength : longueur de l'extension de premier niveau (TLD)
    - HostNoOfSubDomain : nombre de sous-domaines (étiquettes - 2, min 0, hors www)
    - HostIsIP : 1 si l'hôte est une adresse IP, 0 sinon
    - HostNoOfDigits : nombre de chiffres
    - HostDigitRatio : proportion de chiffres
    - HostNoOfLetters : nombre de lettres
    - HostLetterRatio : proportion de lettres
    - HostNoOfHyphens : nombre de tirets '-'
    - HostLongestLabel : longueur de l'étiquette la plus longue
    - HostNoOfOtherSpecial : caractères ni alphanumériques, ni '-', ni '.'
    """
    host = extract_host(url)
    host_len = len(host)
    is_ip = int(_is_ip_address(host))

    labels = host.split(".")

    if is_ip or len(labels) < 2:
        tld_length = 0
        no_of_subdomain = 0
    else:
        tld_length = len(labels[-1])
        no_of_subdomain = max(0, len(labels) - 2)

    n_digits = sum(1 for c in host if c.isdigit())
    digit_ratio = round(n_digits / host_len, 4) if host_len > 0 else 0.0

    n_letters = sum(1 for c in host if c.isalpha())
    letter_ratio = round(n_letters / host_len, 4) if host_len > 0 else 0.0

    n_hyphens = host.count("-")
    longest_label = max((len(lbl) for lbl in labels), default=0)

    return {
        "HostLength": host_len,
        "HostTLDLength": tld_length,
        "HostNoOfSubDomain": no_of_subdomain,
        "HostIsIP": is_ip,
        "HostNoOfDigits": n_digits,
        "HostDigitRatio": digit_ratio,
        "HostNoOfLetters": n_letters,
        "HostLetterRatio": letter_ratio,
        "HostNoOfHyphens": n_hyphens,
        "HostLongestLabel": longest_label,
    }


def host_features_batch(urls: Union[pd.Series, List[str]]) -> pd.DataFrame:
    """
    Calcule les 10 caractéristiques d'hôte pour un ensemble d'URLs.

    Retourne un DataFrame ordonné selon HOST_FEATURES.
    """
    if isinstance(urls, pd.Series):
        url_list = urls.tolist()
    else:
        url_list = list(urls)

    records = [host_features(u) for u in url_list]
    df = pd.DataFrame.from_records(records)
    return df[HOST_FEATURES]
