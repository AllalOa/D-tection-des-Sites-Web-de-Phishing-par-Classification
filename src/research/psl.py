"""
src/research/psl.py - Décomposition de domaine basée sur la Public Suffix List (PSL).

Ce module fournit une décomposition robuste et 100 % hors ligne des noms d'hôtes selon
les spécifications de la Public Suffix List (Mozilla / publicsuffix.org).

Définitions formelles :
- Hostname (FQDN ou IP) : Nom réseau complet de l'hôte (ex: 'subdomain.example.co.uk', '192.168.1.1').
- Public Suffix (eTLD) : Extension sous laquelle un utilisateur peut directement enregistrer un domaine
  (ex: 'com', 'co.uk', 'com.au', 'asso.fr', 'org', 'github.io').
- Registrable Domain (eTLD+1) : Domaine enregistrable = domaine de second niveau + suffixe public
  (ex: 'example.com', 'bbc.co.uk', 'example.com.au').
- Subdomain : Ensemble des étiquettes situées à gauche du domaine enregistrable
  (ex: 'subdomain' dans 'subdomain.example.co.uk', '' dans 'bbc.co.uk').
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple


@dataclass(frozen=True)
class DomainDecomposition:
    """Résultat immuable de la décomposition d'un nom d'hôte."""
    raw_host: str
    hostname: str
    subdomain: str
    domain: str             # Domaine enregistrable (ex: 'bbc.co.uk')
    suffix: str             # Suffixe public (ex: 'co.uk')
    subdomains: Tuple[str, ...]
    no_of_subdomains: int
    is_ip: bool
    ip_version: Optional[int]


# ─────────────────────────────────────────────────────────────────────────────
# Base de données intégrée des suffixes publics majeurs (100% hors-ligne)
# ─────────────────────────────────────────────────────────────────────────────
# Couvre les gTLDs génériques, les ccTLDs à deux lettres et les combinaisons
# de second niveau les plus courantes (Royaume-Uni, Australie, Brésil, Japon, etc.)
_MULTI_LEVEL_SUFFIXES: Set[str] = {
    # Royaume-Uni (UK)
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "net.uk", "ltd.uk", "plc.uk", "sch.uk",
    # Australie (AU)
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    # Nouvelle-Zélande (NZ)
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz", "geek.nz", "gen.nz",
    # Brésil (BR)
    "com.br", "net.br", "org.br", "gov.br", "edu.br", "art.br", "dev.br", "app.br",
    # Japon (JP)
    "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp", "ed.jp", "ad.jp", "gr.jp",
    # Afrique du Sud (ZA)
    "co.za", "org.za", "net.za", "gov.za", "ac.za", "edu.za",
    # Chine & Asie
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.hk", "net.hk", "org.hk", "gov.hk", "edu.hk",
    "com.tw", "org.tw", "gov.tw", "net.tw", "edu.tw",
    "com.sg", "net.sg", "org.sg", "gov.sg", "edu.sg",
    "com.my", "net.my", "org.my", "gov.my", "edu.my",
    # Inde (IN)
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in", "ac.in", "edu.in", "res.in", "gov.in",
    # France (FR)
    "asso.fr", "gouv.fr", "nom.fr", "prd.fr", "presse.fr", "tm.fr", "avocat.fr",
    # Canada (CA)
    "gc.ca", "qc.ca", "on.ca", "bc.ca", "ab.ca", "mb.ca", "sk.ca", "ns.ca", "nb.ca", "nl.ca", "pe.ca",
    # Amérique Latine
    "com.mx", "gob.mx", "edu.mx", "org.mx", "net.mx",
    "com.ar", "gob.ar", "edu.ar", "org.ar", "net.ar",
    "com.co", "gov.co", "edu.co", "org.co",
    "com.cl", "gob.cl", "edu.cl",
    # Turquie, Israël, etc.
    "com.tr", "edu.tr", "gov.tr", "org.tr", "net.tr",
    "co.il", "org.il", "gov.il", "ac.il", "net.il",
    # Plateformes privées courantes
    "github.io", "gitlab.io", "pages.dev", "web.app", "firebaseapp.com",
    "herokuapp.com", "azurewebsites.net", "s3.amazonaws.com", "cloudfront.net",
}


def _is_ip(host: str) -> Tuple[bool, Optional[int]]:
    """Vérifie si host est une adresse IPv4 ou IPv6 valide."""
    clean_host = host.strip("[]")
    try:
        ip = ipaddress.ip_address(clean_host)
        return True, ip.version
    except ValueError:
        return False, None


def decompose_domain(hostname: str) -> DomainDecomposition:
    """
    Décompose un nom d'hôte ou FQDN selon la Public Suffix List.

    Parameters
    ----------
    hostname : str
        Le nom d'hôte brut ou normalisé (ex: 'www.subdomain.example.co.uk', '192.168.1.1').

    Returns
    -------
    DomainDecomposition
        Objet structuré contenant hostname, subdomain, domain, suffix, etc.
    """
    raw_host = str(hostname) if hostname is not None else ""
    cleaned = raw_host.strip().lower()

    # Retirer les éventuels identifiants utilisateur et ports résiduels
    if "@" in cleaned:
        cleaned = cleaned.rsplit("@", 1)[-1]
    if cleaned.startswith("[") and "]" in cleaned:
        # IPv6 avec port éventuel : [2001:db8::1]:8080
        end_bracket = cleaned.index("]")
        cleaned = cleaned[1:end_bracket]
    elif ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0]

    # Retirer le point final éventuel
    cleaned = cleaned.rstrip(".")

    is_ip_addr, ip_ver = _is_ip(cleaned)
    if is_ip_addr:
        return DomainDecomposition(
            raw_host=raw_host,
            hostname=cleaned,
            subdomain="",
            domain=cleaned,
            suffix="",
            subdomains=(),
            no_of_subdomains=0,
            is_ip=True,
            ip_version=ip_ver,
        )

    labels = cleaned.split(".")
    n_labels = len(labels)

    if n_labels <= 1 or not cleaned:
        return DomainDecomposition(
            raw_host=raw_host,
            hostname=cleaned,
            subdomain="",
            domain=cleaned,
            suffix="",
            subdomains=(),
            no_of_subdomains=0,
            is_ip=False,
            ip_version=None,
        )

    # 1. Recherche du suffixe public le plus long (multi-niveaux d'abord)
    matched_suffix = ""
    for k in (3, 2):
        if n_labels >= k:
            candidate = ".".join(labels[-k:])
            if candidate in _MULTI_LEVEL_SUFFIXES:
                matched_suffix = candidate
                break

    # Si aucun multi-niveaux, le suffixe est le dernier label (ex: 'com', 'org', 'fr')
    if not matched_suffix:
        matched_suffix = labels[-1]

    suffix_label_count = len(matched_suffix.split("."))

    # 2. Le domaine enregistrable comprend le suffixe + le label immédiatement à sa gauche
    if n_labels > suffix_label_count:
        registrable_domain = ".".join(labels[-(suffix_label_count + 1):])
        subdomain_labels = tuple(labels[:-(suffix_label_count + 1)])
    else:
        # Cas limite : hôte = suffixe seul (ex: 'co.uk')
        registrable_domain = matched_suffix
        subdomain_labels = ()

    # Nettoyer 'www' des sous-domaines si présent
    subdomain_str = ".".join(subdomain_labels)

    return DomainDecomposition(
        raw_host=raw_host,
        hostname=cleaned,
        subdomain=subdomain_str,
        domain=registrable_domain,
        suffix=matched_suffix,
        subdomains=subdomain_labels,
        no_of_subdomains=len(subdomain_labels),
        is_ip=False,
        ip_version=None,
    )


def get_public_suffix(host: str) -> str:
    """Retourne l'extension publique (ex: 'co.uk', 'com')."""
    return decompose_domain(host).suffix


def get_registrable_domain(host: str) -> str:
    """Retourne le domaine enregistrable (ex: 'bbc.co.uk', 'example.com')."""
    return decompose_domain(host).domain


def count_subdomains_psl(host: str, exclude_www: bool = True) -> int:
    """
    Compte rigoureusement le nombre de sous-domaines via la PSL.

    Parameters
    ----------
    host : str
    exclude_www : bool, default=True
        Si True, l'étiquette 'www' n'est pas comptabilisée comme sous-domaine.
    """
    decomp = decompose_domain(host)
    if decomp.is_ip or not decomp.subdomains:
        return 0
    sub_labels = decomp.subdomains
    if exclude_www:
        sub_labels = tuple(lbl for lbl in sub_labels if lbl != "www")
    return len(sub_labels)
