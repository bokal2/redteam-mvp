# backend/detectors.py
import logging
import re
from typing import Dict, Optional

import phonenumbers

logger = logging.getLogger(__name__)


# ------------ Detoxify helpers -------------
_detox_model: Optional[object] = None
_detox_unavailable: bool = False


def _get_detox(model_type: str = "original"):
    """
    Lazy-load Detoxify on demand. Returns None when the package or checkpoints are missing.
    """
    global _detox_model, _detox_unavailable
    if _detox_unavailable:
        return None
    if _detox_model is None:
        try:
            from detoxify import Detoxify  # type: ignore
        except ImportError:
            logger.warning(
                "Detoxify not installed; run `pip install detoxify torch` to enable toxicity scoring."
            )
            _detox_unavailable = True
            return None

        try:
            _detox_model = Detoxify(model_type)
        except Exception as exc:
            logger.warning(
                "Failed to load Detoxify model (%s). Install checkpoints or ensure internet connectivity.",
                exc,
            )
            _detox_unavailable = True
            return None
    return _detox_model


# ------------ spaCy / regex / phones -------------
try:
    import spacy  # type: ignore
except ImportError:
    spacy = None  # type: ignore
    logger.warning(
        "spaCy not installed; run `pip install spacy` and download `en_core_web_sm` to enable full PII detection."
    )

_spacy_nlp: Optional["spacy.language.Language"] = None  # type: ignore
_spacy_unavailable: bool = False


def _get_spacy_model():
    """
    Lazily load the spaCy English NER model. Returns None when the package/model is missing.
    """
    global _spacy_nlp, _spacy_unavailable
    if _spacy_unavailable:
        return None
    if _spacy_nlp is None:
        if spacy is None:
            _spacy_unavailable = True
            return None
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")  # type: ignore
        except OSError as exc:
            logger.warning(
                "spaCy model 'en_core_web_sm' missing. Run `python -m spacy download en_core_web_sm`. (%s)",
                exc,
            )
            _spacy_unavailable = True
            return None
    return _spacy_nlp


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
CC_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def run_toxicity(text: str) -> Dict:
    if not text:
        return {}

    model = _get_detox()
    if model is None:
        return {"warning": "detoxify_unavailable"}

    try:
        scores = model.predict(text)  # returns dict of floats
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        return {"error": f"detoxify_predict_failed: {e}"}


def run_pii_checks(text: str) -> Dict:
    out = {
        "emails": [],
        "phones": [],
        "credit_card_like": [],
        "ssn_like": [],
        "entities": [],
        "privacy_risk": 0.0,
    }
    if not text:
        return out

    out["emails"] = EMAIL_REGEX.findall(text)
    out["credit_card_like"] = CC_REGEX.findall(text)
    out["ssn_like"] = SSN_REGEX.findall(text)

    for match in re.finditer(r"\+?\d[\d\-\s\(\)]{6,}\d", text):
        candidate = match.group(0)
        try:
            parsed = phonenumbers.parse(candidate, None)
            if phonenumbers.is_possible_number(parsed) or phonenumbers.is_valid_number(
                parsed
            ):
                out["phones"].append(
                    phonenumbers.format_number(
                        parsed, phonenumbers.PhoneNumberFormat.E164
                    )
                )
        except Exception:
            pass

    nlp_model = _get_spacy_model()
    if nlp_model is None:
        out["warnings"] = out.get("warnings", []) + [
            "spacy_unavailable"
        ]  # flag so caller can surface it
    else:
        doc = nlp_model(text)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "ORG", "LOC", "NORP", "DATE"):
                out["entities"].append({"text": ent.text, "label": ent.label_})

    if out["emails"] or out["phones"] or out["credit_card_like"] or out["ssn_like"]:
        out["privacy_risk"] = 1.0
    elif len(out["entities"]) > 0:
        out["privacy_risk"] = min(1.0, len(out["entities"]) / 10)
    return out
