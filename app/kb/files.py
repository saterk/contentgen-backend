"""Structured KB files that are loaded in full, not vectorised (spec section 6.1).

taxonomy.json, compliance/disclaimers.json, compliance/banned_terms.json,
brand/brand_guide.md, compliance/image_rules.md and manifest.json.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

TAXONOMY_FIELDS = ("style", "tone", "format")  # the content profile
_PLURAL = {"style": "styles", "tone": "tones", "format": "formats", "visual_style": "visual_styles"}


def norm_text(s: str) -> str:
    """Lowercase, unify dashes and collapse whitespace/punctuation (keeps Thai characters)."""
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w฀-๿%+-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_term(text: str, term: str) -> bool:
    """Whole-word match for latin terms, substring match for Thai (no word spaces)."""
    t, n = norm_text(term), norm_text(text)
    if not t:
        return False
    if t.isascii():
        return re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", n) is not None
    return t in n


@dataclass
class Taxonomy:
    # field -> {value -> [aliases]}
    values: dict[str, dict[str, list[str]]]

    def allowed(self, field_name: str) -> list[str]:
        return list(self.values.get(field_name, {}).keys())

    def normalise(self, field_name: str, raw: str | None) -> str | None:
        """Map free text to an allowed value, or None if nothing fits.

        Order: exact value or alias → value/alias mentioned in the text (first in
        taxonomy order wins, e.g. "friendly and warm" → "warm") → close spelling.
        """
        if not raw:
            return None
        options = self.values.get(field_name, {})
        text = norm_text(raw)
        for value, aliases in options.items():
            if text == norm_text(value) or text in {norm_text(a) for a in aliases}:
                return value
        for value, aliases in options.items():
            if any(contains_term(text, t) for t in [value.replace("_", " "), *aliases]):
                return value
        lookup = {norm_text(v): v for v in options}
        for v, aliases in options.items():
            for a in aliases:
                lookup.setdefault(norm_text(a), v)
        close = difflib.get_close_matches(text, list(lookup), n=1, cutoff=0.8)
        return lookup[close[0]] if close else None


@dataclass
class Disclaimer:
    id: str
    product_id: str
    text: str


@dataclass
class KBFiles:
    root: Path
    taxonomy: Taxonomy
    disclaimers: list[Disclaimer]
    banned_terms: list[str]
    brand_guide: str
    image_rules: str
    manifest: dict = field(default_factory=dict)

    @property
    def kb_version(self) -> str:
        return str(self.manifest.get("kb_version", "unversioned"))

    def disclaimers_for(self, product_id: str) -> list[Disclaimer]:
        return [d for d in self.disclaimers if d.product_id == product_id]

    def disclaimer(self, disclaimer_id: str) -> Disclaimer | None:
        return next((d for d in self.disclaimers if d.id == disclaimer_id), None)

    def _guide_section(self, heading: str) -> str:
        m = re.search(rf"^##\s*{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)", self.brand_guide, re.M | re.S | re.I)
        return " ".join(m.group(1).split()) if m else ""

    def image_style_suffix(self, visual_style: str | None = None) -> str:
        """Brand guide '## Visual style: <name>' text plus the common '## Image style suffix'."""
        parts = [self._guide_section(f"Visual style: {visual_style}") if visual_style else "",
                 self._guide_section("Image style suffix")]
        return ", ".join(p for p in parts if p)

    def reload_manifest(self) -> None:
        self.manifest = _read_json(self.root / "manifest.json", default={})

    @classmethod
    def load(cls, kb_dir: Path) -> "KBFiles":
        tax_raw = _read_json(kb_dir / "taxonomy.json", default={})
        taxonomy = Taxonomy(values={f: dict(tax_raw.get(plural, {})) for f, plural in _PLURAL.items()})
        disclaimers = [
            Disclaimer(id=d["id"], product_id=d["product_id"], text=d["text"])
            for d in _read_json(kb_dir / "compliance" / "disclaimers.json", default=[])
        ]
        banned = _read_json(kb_dir / "compliance" / "banned_terms.json", default={"terms": []})
        return cls(
            root=kb_dir,
            taxonomy=taxonomy,
            disclaimers=disclaimers,
            banned_terms=list(banned.get("terms", [])),
            brand_guide=_read_text(kb_dir / "brand" / "brand_guide.md"),
            image_rules=_read_text(kb_dir / "compliance" / "image_rules.md"),
            manifest=_read_json(kb_dir / "manifest.json", default={}),
        )


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
