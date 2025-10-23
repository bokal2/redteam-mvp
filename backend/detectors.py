# backend/detectors.py
import logging
import re
import phonenumbers
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _load_detoxify_safely(model_type: str = "original"):
    """Best-effort Detoxify loader that surfaces helpful errors without crashing."""
    try:
        from detoxify import Detoxify  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "detoxify package is not installed. Run `pip install detoxify` to enable toxicity detection."
        ) from exc

    try:
        return Detoxify(model_type)
    except Exception as exc:
        # Provide actionable guidance but let caller decide on fallback.
        raise RuntimeError(
            "Failed to load Detoxify model. Ensure internet connectivity or manually download the checkpoints. "
            "Install instructions: https://github.com/unitaryai/detoxify"
        ) from exc


_detox_model: Optional[object] = None
_detox_unavailable: bool = False


def _get_detox():
    global _detox_model, _detox_unavailable
    if _detox_unavailable:
        return None
    if _detox_model is None:
        try:
            _detox_model = _load_detoxify_safely("original")
        except Exception as exc:
            logger.warning("Detoxify unavailable: %s", exc)
            _detox_unavailable = True
            return None
    return _detox_model


# ------------ spaCy / regex / phones -------------
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Helpful error if model not installed
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
    )

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

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "GPE", "ORG", "LOC", "NORP", "DATE"):
            out["entities"].append({"text": ent.text, "label": ent.label_})

    if out["emails"] or out["phones"] or out["credit_card_like"] or out["ssn_like"]:
        out["privacy_risk"] = 1.0
    elif len(out["entities"]) > 0:
        out["privacy_risk"] = min(1.0, len(out["entities"]) / 10)
    return out
