"""
src/research/url_parser.py - Analyseur d'URL robuste et conforme (Membre 4 - Fateh).

Fournit :
- Parsing robuste des URLs HTTP, HTTPS et sans schéma.
- Gestion native des adresses IPv4 et IPv6 (avec ou sans crochets, avec ou sans port).
- Gestion des composants d'authentification (user:password@host).
- Intégration de la décomposition PSL (Public Suffix List).
- Validation stricte respectant les spécifications de chaque expérience.
- Élimination formelle de tout découpage manuel par les deux-points (':') sur les hôtes.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from src.config import MAX_URL_LENGTH
from src.research.psl import DomainDecomposition, decompose_domain


class InvalidURLError(ValueError):
    """Exception levée exclusivement pour une entrée URL invalide."""
    pass


@dataclass(frozen=True)
class ParsedURL:
    """Structure immuable représentant une URL analysée de façon robuste."""
    raw_url: str
    normalized_url: str
    scheme: str
    netloc: str
    host: str
    port: Optional[int]
    userinfo: str
    path: str
    query: str
    fragment: str
    is_https: bool
    is_ip: bool
    ip_version: Optional[int]
    domain_decomposition: DomainDecomposition

    @property
    def public_suffix(self) -> str:
        return self.domain_decomposition.suffix

    @property
    def registrable_domain(self) -> str:
        return self.domain_decomposition.domain

    @property
    def subdomain(self) -> str:
        return self.domain_decomposition.subdomain

    @property
    def no_of_subdomains(self) -> int:
        return self.domain_decomposition.no_of_subdomains


def parse_url_robust(url: str, default_scheme: str = "") -> ParsedURL:
    """
    Analyse une URL de manière exhaustive et robuste sans division manuelle de colonnes.

    Gère :
    - HTTP, HTTPS, ou schémas absents.
    - Adresses IPv4 et adresses littérales IPv6 entre crochets [::1]:8080.
    - Ports et identifiants de connexion (credentials).
    - Chemins, requêtes et fragments.

    Parameters
    ----------
    url : str
        L'URL brute en entrée.
    default_scheme : str, default=""
        Schéma à supposer si aucun schéma n'est présent.

    Returns
    -------
    ParsedURL
        Objet structuré contenant tous les attributs validés.

    Raises
    ------
    InvalidURLError
        Si l'URL est de type invalide ou vide.
    """
    if url is None or not isinstance(url, str):
        raise InvalidURLError(f"L'URL doit être une chaîne de caractères, reçu : {type(url).__name__}")

    s = url.strip()
    if not s:
        raise InvalidURLError("L'URL ne peut pas être vide ou contenir uniquement des espaces.")

    if len(s) > MAX_URL_LENGTH:
        raise InvalidURLError(
            f"L'URL dépasse la longueur maximale autorisée ({MAX_URL_LENGTH} caractères). "
            f"Longueur reçue : {len(s)}."
        )

    # Détection de la présence d'un schéma explicite
    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", s))
    if not has_scheme:
        if s.startswith("//"):
            split_target = s
        else:
            # Traiter comme hôte/chemin sans schéma
            split_target = f"//{s}"
    else:
        split_target = s

    try:
        sp = urlsplit(split_target)
    except Exception as exc:
        raise InvalidURLError(f"Impossible de parser l'URL '{url}' : {exc}") from exc

    scheme = sp.scheme.lower() if sp.scheme else default_scheme.lower()
    netloc = sp.netloc or ""
    path = sp.path or ""
    query = sp.query or ""
    fragment = sp.fragment or ""

    # Extraction sécurisée de l'hôte et du port sans découpage de colon
    raw_hostname = sp.hostname or ""
    port = sp.port

    # Extraction des identifiants (username / password)
    userinfo = ""
    if sp.username or sp.password:
        userinfo = f"{sp.username or ''}:{sp.password or ''}"

    # Détection et normalisation de l'adresse IP
    is_ip = False
    ip_version = None
    clean_host = raw_hostname.strip("[]")
    if clean_host:
        try:
            ip_obj = ipaddress.ip_address(clean_host)
            is_ip = True
            ip_version = ip_obj.version
            clean_host = str(ip_obj)
        except ValueError:
            is_ip = False

    # Décomposition PSL
    decomp = decompose_domain(clean_host)

    # Reconstruction de l'URL normalisée
    norm_netloc = netloc
    norm_scheme = scheme if scheme else ("https" if sp.scheme == "https" else "")
    normalized_url = urlunsplit((norm_scheme, norm_netloc, path, query, fragment))

    return ParsedURL(
        raw_url=url,
        normalized_url=normalized_url,
        scheme=scheme,
        netloc=netloc,
        host=clean_host.lower(),
        port=port,
        userinfo=userinfo,
        path=path,
        query=query,
        fragment=fragment,
        is_https=(scheme == "https"),
        is_ip=is_ip,
        ip_version=ip_version,
        domain_decomposition=decomp,
    )


def validate_url(url: str, experiment: Optional[str] = None, allow_schemeless: bool = False) -> ParsedURL:
    """
    Valide l'URL en entrée selon le niveau de tolérance requis.

    Règles :
    - Chaîne non vide <= MAX_URL_LENGTH.
    - Hôte valide requis.
    - Pour Expérience D (ou allow_schemeless=True) : les noms d'hôtes sans schéma sont acceptés.
    - Pour les autres expériences (C, A, B, B') : un schéma valide et un netloc sont strictement requis.

    Parameters
    ----------
    url : str
    experiment : str, optional
    allow_schemeless : bool, default=False

    Returns
    -------
    ParsedURL

    Raises
    ------
    InvalidURLError
    """
    parsed = parse_url_robust(url)

    # Si l'expérience cible est D ou si l'on tolère explicitement les entrées sans schéma
    is_host_only_exp = (experiment == "D" or allow_schemeless)

    if not parsed.host:
        raise InvalidURLError(
            f"L'URL '{url}' ne contient pas d'hôte valide (netloc absent). "
            "Vérifiez le format ou le schéma (ex: 'https://')."
        )

    if not is_host_only_exp and not parsed.scheme:
        raise InvalidURLError(
            f"L'URL '{url}' ne contient pas d'hôte valide (netloc absent). "
            "Vérifiez le schéma (ex: 'https://')."
        )

    return parsed
