"""Normalization applied to text on the way in, before it is hashed, counted or split."""

# Arabic-Indic (U+0660-U+0669) and Persian/Urdu Extended Arabic-Indic (U+06F0-U+06F9)
# digits, mapped onto the ASCII digits they stand for. Both blocks are ordered 0-9,
# so the mapping is positional.
_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_ASCII_DIGITS = "0123456789"

_DIGITS: dict[int, str] = str.maketrans(
    _ARABIC_INDIC + _EXTENDED_ARABIC_INDIC,
    _ASCII_DIGITS + _ASCII_DIGITS,
)


def sanitize(text: str) -> str:
    """Rewrite text into the form that gets stored and chunked.

    Today that is digits only: ١٣ and ۱۳ both become 13, so a number reads the same
    to the splitter, the token counter and the embedding model whichever script it
    arrived in. Nothing else about the text is touched.
    """
    return text.translate(_DIGITS)
