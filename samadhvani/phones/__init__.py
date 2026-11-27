"""Phone layer: the articulatory inventory and Devanāgarī -> phone mapping."""

from .inventory import (
    PHONES,
    STOBHAS,
    VOWEL_FORMANTS,
    DIPHTHONGS,
    CONSONANT_LOCUS,
    NOISE_BANDS,
    Phone,
    phone,
    all_phones,
)
from .g2p import (
    AksaraPhones,
    aksara_to_phones,
    text_to_phone_string,
    homorganic_anusvara,
    resolve_visarga,
)

__all__ = [
    "PHONES",
    "STOBHAS",
    "VOWEL_FORMANTS",
    "DIPHTHONGS",
    "CONSONANT_LOCUS",
    "NOISE_BANDS",
    "Phone",
    "phone",
    "all_phones",
    "AksaraPhones",
    "aksara_to_phones",
    "text_to_phone_string",
    "homorganic_anusvara",
    "resolve_visarga",
]
