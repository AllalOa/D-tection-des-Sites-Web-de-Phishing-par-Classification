"""
url_features.py - Calcul des variables URL_ONLY depuis la chaîne d'URL seule.

Aucun accès réseau. Calcule les 18 variables URL_ONLY depuis la chaîne brute.
Utilisé pour l'expérience C (modèle déployable) et pour predict.py.

Fonctions exportées :
    compute_url_features(url: str) -> dict
    compute_url_features_batch(urls: list | pd.Series) -> pd.DataFrame

Variables calculées (18) :
    URLLength, DomainLength, IsDomainIP, TLDLength, NoOfSubDomain,
    HasObfuscation, NoOfObfuscatedChar, ObfuscationRatio,
    NoOfLettersInURL, LetterRatioInURL, NoOfDegitsInURL, DegitRatioInURL,
    NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL,
    NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL, IsHTTPS
"""

import re
import math
import ipaddress
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd

from src.config import URL_ONLY_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# Expressions régulières précompilées
# ─────────────────────────────────────────────────────────────────────────────

# Caractères de ponctuation "spéciaux" de l'URL (hors lettres, chiffres, /, :, ., -)
_RE_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9/:.\-]")

# Séquences suspectes d'obfuscation (encodage %, répétition de caractères spéciaux)
_RE_PERCENT_ENCODING = re.compile(r"%[0-9a-fA-F]{2}")

# Détecter si le host est une adresse IP
_RE_IPV4 = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)

# Caractères considérés comme "spéciaux" pour SpacialCharRatioInURL
# (hors alphanumériques, sans les séparateurs URL standard / : . -)
_SPECIAL_CHARS = set("@!#$%^&*()+=[]{}|;',<>?~`\"\\")


def _is_ip_address(host: str) -> bool:
    """Retourne True si host est une adresse IPv4 ou IPv6 valide."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def compute_url_features(url: str) -> dict:
    """
    Calcule les 18 variables URL_ONLY depuis la chaîne d'URL brute.

    Aucun accès réseau. Utilise uniquement les opérations sur string.

    Parameters
    ----------
    url : str
        L'URL brute (ex: 'https://www.example.com/path?q=1').

    Returns
    -------
    dict
        Dictionnaire {nom_variable: valeur} pour les 18 variables.
    """
    # Garantir que url est une chaîne
    url = str(url) if not isinstance(url, str) else url

    # ── Parsing de l'URL ───────────────────────────────────────────────
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc or ""
        # Supprimer le port s'il est présent
        host = netloc.split(":")[0] if netloc else ""
    except Exception:
        scheme = ""
        netloc = ""
        host = ""

    # ── 1. URLLength ───────────────────────────────────────────────────
    url_length = len(url)

    # ── 2. IsHTTPS ────────────────────────────────────────────────────
    is_https = int(scheme == "https")

    # ── 3. IsDomainIP ─────────────────────────────────────────────────
    is_domain_ip = int(_is_ip_address(host))

    # ── 4. DomainLength ───────────────────────────────────────────────
    domain_length = len(host)

    # ── 5. TLD et TLDLength ───────────────────────────────────────────
    # TLD = dernière partie du host après le dernier point
    # Si IP ou vide → TLD = ""
    if host and not is_domain_ip and "." in host:
        tld = host.rsplit(".", 1)[-1]
    else:
        tld = ""
    tld_length = len(tld)

    # ── 6. NoOfSubDomain ──────────────────────────────────────────────
    # Nombre de sous-domaines = nombre de points dans host - 1
    # ex: www.sub.example.com → 2 sous-domaines (www, sub)
    if host and not is_domain_ip and host.count(".") >= 2:
        no_of_subdomain = host.count(".") - 1
    else:
        no_of_subdomain = 0

    # ── 7. Obfuscation (encodage %) ───────────────────────────────────
    # Cherche les séquences %xx dans l'URL complète
    percent_encoded = _RE_PERCENT_ENCODING.findall(url)
    no_of_obfuscated = len(percent_encoded)
    has_obfuscation = int(no_of_obfuscated > 0)
    obfuscation_ratio = no_of_obfuscated / url_length if url_length > 0 else 0.0

    # ── 8. Lettres dans l'URL ─────────────────────────────────────────
    no_of_letters = sum(1 for c in url if c.isalpha())
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0.0

    # ── 9. Chiffres dans l'URL ────────────────────────────────────────
    no_of_digits = sum(1 for c in url if c.isdigit())
    digit_ratio = no_of_digits / url_length if url_length > 0 else 0.0

    # ── 10. Caractères spéciaux de formulaire ─────────────────────────
    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    # ── 11. Autres caractères spéciaux ───────────────────────────────
    # Caractères qui ne sont pas lettres, chiffres, ni : / . - ? = & #
    standard_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "0123456789/:.-?=&#_~")
    no_of_other_special = sum(1 for c in url if c not in standard_chars)

    # ── 12. SpacialCharRatioInURL ─────────────────────────────────────
    # Ratio de caractères parmi _SPECIAL_CHARS dans l'URL
    no_spacial = sum(1 for c in url if c in _SPECIAL_CHARS)
    spacial_char_ratio = no_spacial / url_length if url_length > 0 else 0.0

    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special,
        "SpacialCharRatioInURL": spacial_char_ratio,
        "IsHTTPS": is_https,
    }


def compute_url_features_batch(
    urls,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calcule les features URL_ONLY pour une liste ou Series d'URLs.

    Parameters
    ----------
    urls : list or pd.Series
        Collection d'URLs.
    verbose : bool
        Affiche la progression si True.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les 18 colonnes URL_ONLY_FEATURES, indexé comme urls.
    """
    if isinstance(urls, pd.Series):
        url_list = urls.tolist()
        index = urls.index
    else:
        url_list = list(urls)
        index = None

    records = []
    total = len(url_list)
    for i, url in enumerate(url_list):
        if verbose and i % 10_000 == 0:
            print(f"  [url_features] {i}/{total}...")
        records.append(compute_url_features(url))

    df_result = pd.DataFrame(records)
    if index is not None:
        df_result.index = index

    # S'assurer que toutes les colonnes sont présentes dans le bon ordre
    for col in URL_ONLY_FEATURES:
        if col not in df_result.columns:
            df_result[col] = 0

    return df_result[URL_ONLY_FEATURES]
