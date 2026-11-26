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
from urllib.parse import urlparse, urlsplit, unquote

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

# Mots-clés fréquemment rencontrés dans les URLs de phishing (ingénierie sociale / spoofing)
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "log-in", "sign-in", "verify", "verification",
    "account", "update", "banking", "secure", "security", "webscr",
    "ebayisapi", "paypal", "appleid", "wallet", "confirm", "password",
    "credential", "auth", "support", "billing", "recover", "service",
    "portal", "validate", "token", "session"
]

# Principaux services de réduction d'URLs souvent abusés
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly"
}


def calculate_entropy(text: str) -> float:
    """Calcule l'entropie de Shannon d'une chaîne de caractères."""
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum(p * math.log2(p) for p in prob)


def extract_lexical_descriptors(url: str) -> dict:
    """
    Extrait l'ensemble des descripteurs lexicaux d'une URL pour l'analyse en temps réel.
    (longueur, présence d'IP, nb de caractères spéciaux, sous-domaines, HTTPS, mots suspects, entropie...)
    """
    url_str = str(url) if not isinstance(url, str) else url
    try:
        parsed = urlsplit(url_str)
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        raw_host = parsed.hostname or ""
        host = raw_host.strip("[]").lower() if raw_host else ""
        path = parsed.path or ""
    except Exception:
        scheme = ""
        host = ""
        path = ""

    url_lower = url_str.lower()
    is_domain_ip = int(_is_ip_address(host))
    subdomains_count = max(0, host.count(".") - 1) if (host and not is_domain_ip) else 0

    # Recherche des mots suspects
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]

    return {
        "url": url_str,
        "url_length": len(url_str),
        "domain": host,
        "domain_length": len(host),
        "path_length": len(path),
        "is_https": int(scheme == "https"),
        "is_domain_ip": is_domain_ip,
        "nb_subdomains": subdomains_count,
        "nb_special_chars": sum(1 for c in url_str if c in _SPECIAL_CHARS),
        "nb_hyphens": host.count("-"),
        "nb_at_symbol": url_str.count("@"),
        "nb_dots_in_host": host.count("."),
        "nb_suspicious_keywords": len(found_keywords),
        "suspicious_keywords_found": found_keywords,
        "is_shortened": int(host in URL_SHORTENERS),
        "shannon_entropy": round(calculate_entropy(url_str), 4),
    }


def _is_ip_address(host: str) -> bool:
    """Retourne True si host est une adresse IPv4 ou IPv6 valide."""
    if not host:
        return False
    try:
        clean_host = host.strip("[]")
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def compute_url_features(url: str, extended: bool = False) -> dict:
    """
    Calcule les 18 variables URL_ONLY depuis la chaîne d'URL brute.

    Aucun accès réseau. Utilise uniquement les opérations sur string.

    Parameters
    ----------
    url : str
        L'URL brute (ex: 'https://www.example.com/path?q=1').
    extended : bool, optional
        Si True, inclut également un sous-dictionnaire 'lexical_descriptors' enrichi.

    Returns
    -------
    dict
        Dictionnaire {nom_variable: valeur} pour les 18 variables (+ descripteurs si extended).
    """
    # Garantir que url est une chaîne
    url = str(url) if not isinstance(url, str) else url

    # ── Parsing de l'URL ───────────────────────────────────────────────
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        raw_host = parsed.hostname or ""
        host = raw_host.strip("[]").lower() if raw_host else ""
    except Exception:
        scheme = ""
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

    features_dict = {
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
    if extended:
        features_dict["lexical_descriptors"] = extract_lexical_descriptors(url)
    return features_dict


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
