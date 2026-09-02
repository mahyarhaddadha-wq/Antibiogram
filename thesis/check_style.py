#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the thesis chapters against the agreed writing rules.

  python3 thesis/check_style.py
"""
import re
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parent
KASRA = "ِ"          # ARABIC KASRA, the ezafe diacritic to be removed
DASHES = "—–"   # em dash, en dash

# Module-number references the text must not use (rule 10).
MODULE_RE = re.compile(r"ماژول\s*[۰-۹\d]")

def strip_structure(text):
    """Drop headings, tables, code and footnote definitions before prose checks."""
    out = []
    fenced = False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            fenced = not fenced
            continue
        if fenced or s.startswith("#") or s.startswith("|") or s.startswith("[^"):
            continue
        # English-only lines (keyword lists, the English abstract) follow
        # English typography, so they are not checked for Persian style.
        if s and not re.search(r"[\u0600-\u06FF]", s):
            continue
        out.append(line)
    return "\n".join(out)

def sentences(prose):
    """Split declarative prose into sentences, ignoring list markers."""
    plain = re.sub(r"\*\*?|`[^`]*`|\$[^$]*\$", "", prose)
    plain = re.sub(r"^\s*[-•]\s*", "", plain, flags=re.M)
    return [s.strip() for s in re.split(r"[.؟!]\s|\n\n", plain) if s.strip()]

def main():
    files = sorted(THESIS.glob("chapter_0*.md")) + [THESIS / "front_matter.md"]
    total = {"kasra": 0, "dash": 0, "underscore": 0, "q": 0, "excl": 0, "module": 0, "latex": 0}
    bad = False

    for f in files:
        if not f.exists():
            continue
        raw = f.read_text(encoding="utf-8")
        prose = strip_structure(raw)

        counts = {
            "kasra": raw.count(KASRA),
            "dash": sum(prose.count(d) for d in DASHES),
            "underscore": len(re.findall(r"(?<!`)_(?!`)", prose)),
            "q": prose.count("؟"),
            "excl": prose.count("!"),
            "module": len(MODULE_RE.findall(prose)),
            # Inline math is copied into the document verbatim, so a LaTeX
            # macro left inside it prints as "\\omega_0" on the page.
            "latex": len(re.findall(r"(?<!\$)\$(?!\$)[^$\n]*\\[A-Za-z]", raw)),
        }
        for k, v in counts.items():
            total[k] += v
        flag = any(counts[k] for k in ("kasra", "dash", "module", "latex"))
        bad = bad or flag
        mark = "FAIL" if flag else "ok  "
        print(f"  [{mark}] {f.name:22s} " + "  ".join(f"{k}={v}" for k, v in counts.items()))

        if counts["module"]:
            for m in MODULE_RE.finditer(prose):
                print(f"           module ref: ...{prose[max(0,m.start()-40):m.start()+25]}...")

        for m in re.finditer(r"(?<!\$)\$(?!\$)[^$\n]*\\[A-Za-z][^$\n]*\$", raw):
            print(f"           latex left in inline math: {m.group(0)}")

    print("\n  TOTAL " + "  ".join(f"{k}={v}" for k, v in total.items()))
    if total["q"] > 1:
        print(f"  NOTE: {total['q']} question marks; at most one rhetorical question was agreed.")
    print("\n  " + ("Style check FAILED." if bad else "Style check passed."))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
