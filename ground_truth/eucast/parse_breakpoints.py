"""
استخراجِ جدولِ نقاطِ شکستِ EUCAST v16.0 به یک فایلِ ماشین‌خوان.

ورودی : v_16.0_Breakpoint_Tables.pdf (نسخه‌ی رسمی، همان که در ریپازیتوری است)
خروجی : ground_truth/eucast/eucast_v16_zone_breakpoints.csv

چرا استخراجِ مختصات‌محور و نه متنِ خام: در جریانِ متنیِ PDF، نامِ عامل و اعداد پشتِ
هم می‌آیند و ستون‌ها از هم قابلِ تفکیک نیستند (سلولِ خالی اصلاً چیزی تولید نمی‌کند،
پس شمارشِ اعداد جواب نمی‌دهد). ولی PDF مختصاتِ هر کلمه را دارد و ستون‌ها در این سند
هم‌ترازِ دقیق‌اند، پس تخصیصِ ستون بر اساسِ x قطعی است.

ساختارِ هر سطر در سندِ اصلی:

    نامِ عامل | MIC S≤ | MIC R> | MIC ATU | محتوایِ دیسک (µg) | zone S≥ | zone R< | zone ATU

ما فقط سه ستونِ آخر به‌علاوه‌ی محتوایِ دیسک را می‌خواهیم (روشِ ما دیسک‌دیفیوژن است).

قرارداد EUCAST برایِ تفسیر:
    قطر ≥ S      -> S  (حساس، رژیمِ استاندارد)
    قطر < R      -> R  (مقاوم)
    R ≤ قطر < S  -> I  (حساس با مواجهه‌ی افزایش‌یافته)
اگر S و R برابر باشند دسته‌ی I وجود ندارد (بیشترِ عامل‌ها همین‌طورند).
ATU (ناحیه‌ی عدمِ قطعیتِ فنی) یک بازه‌ی جداست که در آن EUCAST می‌گوید نتیجه را
نباید گزارش کرد -- برایِ ما مهم است چون دقیقاً همان‌جایی است که خطایِ اندازه‌گیری
بیشترین اثر را دارد.
"""
import csv
import re
import sys
from pathlib import Path

import pymupdf

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "v_16.0_Breakpoint_Tables.pdf"
OUT = REPO / "ground_truth" / "eucast" / "eucast_v16_zone_breakpoints.csv"

# مرزهایِ ستون بر حسبِ x، از هم‌ترازیِ سرستون‌هایِ خودِ سند خوانده شده.
# سرستون‌ها: MIC S≤ ~۱۹۰، MIC R> ~۲۲۵، MIC ATU ~۲۵۹، دیسک ~۳۰۰، zone S≥ ~۳۳۸،
#            zone R< ~۳۷۴، zone ATU ~۴۰۸
COLS = [
    ("agent",     0.0,   180.0),
    ("mic_s",   180.0,   215.0),
    ("mic_r",   215.0,   250.0),
    ("mic_atu", 250.0,   285.0),
    ("disk_ug", 285.0,   330.0),
    ("zone_s",  330.0,   365.0),
    ("zone_r",  365.0,   400.0),
    ("zone_atu", 400.0,  440.0),
]

ROW_TOL = 3.0          # پیکسل: کلماتی که مرکزِ عمودی‌شان تا این حد فاصله دارد، یک سطرند
HEADER_Y = 125.0       # زیرِ نوارِ سرستون‌ها
FOOTER_MARK = re.compile(r"^\s*\d+\.\s")   # سطرهایِ پانویسِ شماره‌دار

# نشانه‌هایی که مقدارِ عددی نیستند و باید عیناً حفظ شوند
NON_NUMERIC = {"-", "IE", "NA", "Note"}


# صفحه‌هایی که سرستونِ جدول را دارند ولی ارگانیسم نیستند
NOT_AN_ORGANISM = ("Guidance on reading", "Dosages", "PK-PD", "Notes")


def page_organism(page):
    """نامِ گونه/گروهِ ارگانیسم از عنوانِ بالا-چپِ صفحه.

    فقط **بالاترین خطِ** بالا-چپ گرفته می‌شود: خطِ دومِ همان ناحیه در سند
    «Expert Rules and Expected Phenotypes» است که بخشی از نام نیست.
    """
    words = [w for w in page.get_text("words") if w[1] < 75 and w[0] < 300]
    if not words:
        return None
    top = min(w[1] for w in words)
    line = [w for w in words if abs(w[1] - top) <= ROW_TOL]
    line.sort(key=lambda w: w[0])
    name = " ".join(w[4] for w in line).replace("*", "").strip()
    name = re.sub(r"\s+", " ", name)
    if not name or any(name.startswith(p) for p in NOT_AN_ORGANISM):
        return None
    return name


def is_disk_diffusion_page(page):
    """صفحه‌ای که واقعاً جدولِ نقاطِ شکست دارد، هر دو سرستونِ «S ≥» و «R <» را دارد."""
    txt = page.get_text()
    return ("S ≥" in txt or "S≥" in txt) and ("R <" in txt or "R<" in txt)


def rows_from_page(page):
    """کلماتِ بدنه‌ی جدول را به سطر و سپس به ستون تقسیم می‌کند."""
    words = [w for w in page.get_text("words") if w[1] > HEADER_Y]
    if not words:
        return []
    words.sort(key=lambda w: (w[1], w[0]))

    lines, cur, cur_y = [], [], None
    for w in words:
        yc = 0.5 * (w[1] + w[3])
        if cur_y is None or abs(yc - cur_y) <= ROW_TOL:
            cur.append(w)
            cur_y = yc if cur_y is None else (cur_y + yc) / 2.0
        else:
            lines.append(cur)
            cur, cur_y = [w], yc
    if cur:
        lines.append(cur)

    out = []
    for ln in lines:
        cells = {name: [] for name, _, _ in COLS}
        for w in ln:
            xc = 0.5 * (w[0] + w[2])
            for name, lo, hi in COLS:
                if lo <= xc < hi:
                    cells[name].append((w[0], w[4]))
                    break
        rec = {}
        for name, _, _ in COLS:
            parts = [t for _, t in sorted(cells[name])]
            rec[name] = " ".join(parts).strip()
        out.append((ln[0][1], rec))
    return out


def clean_value(s):
    """عددِ نقطه‌ی شکست را از حاشیه‌نویسیِ سند جدا می‌کند.

    در سند، شماره‌ی پانویس بدونِ فاصله به عدد می‌چسبد ('14A'، '82'، '(19)A,D').
    پانویسِ حرفی همیشه بعدِ عدد می‌آید، پس با یک الگویِ عددیِ ابتدای‌رشته جدا می‌شود.
    پرانتز در EUCAST یعنی «نقطه‌ی شکست با احتیاط» -- نگه داشته می‌شود تا از دست نرود.
    """
    s = (s or "").strip()
    if not s:
        return "", ""
    for tok in NON_NUMERIC:
        if s.startswith(tok):
            return tok, s[len(tok):].strip()
    m = re.match(r"^\(?\s*(\d+(?:\.\d+)?)\s*\)?", s)
    if not m:
        return "", s
    return m.group(1), s[m.end():].strip()


def main():
    if not PDF.exists():
        sys.exit(f"جدولِ EUCAST پیدا نشد: {PDF}")
    doc = pymupdf.open(PDF)

    rows = []
    for idx in range(len(doc)):
        page = doc[idx]
        if not is_disk_diffusion_page(page):
            continue
        org = page_organism(page)
        if not org:
            continue
        section = None
        last = None          # آخرین سطرِ افزوده‌شده از همین صفحه، برایِ چسباندنِ ادامه‌ی نام
        for _, rec in rows_from_page(page):
            agent = rec["agent"].strip()
            if not agent:
                last = None
                continue
            if FOOTER_MARK.match(agent):
                break                       # از این‌جا به بعد پانویس است، نه جدول
            zs, _ = clean_value(rec["zone_s"])
            zr, _ = clean_value(rec["zone_r"])
            if not zs and not zr:
                has_other = any(rec[c] for c in ("mic_s", "mic_r", "disk_ug"))
                # نامِ بلندِ عامل در سند به خطِ بعد می‌شکند. خطِ شکسته هیچ عددی ندارد و
                # بلافاصله بعدِ سطرِ اصلی می‌آید، پس به نامِ همان سطر چسبانده می‌شود --
                # وگرنه نامِ عامل بریده ذخیره می‌شود و جدول برایِ جست‌وجو بی‌فایده است.
                if last is not None and not has_other:
                    last["agent"] = f"{last['agent']} {agent}".strip()
                    continue
                if len(agent) < 40 and not has_other:
                    section = agent
                last = None
                continue
            disk, _ = clean_value(rec["disk_ug"])
            atu = rec["zone_atu"].strip()
            last = {
                "organism": org,
                "agent_class": section or "",
                "agent": agent,
                "disk_content_ug": rec["disk_ug"].strip(),
                "zone_S_ge_mm": zs,
                "zone_R_lt_mm": zr,
                "zone_ATU": atu,
                "mic_S_le": rec["mic_s"].strip(),
                "mic_R_gt": rec["mic_r"].strip(),
                "pdf_page": idx + 1,
            }
            rows.append(last)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    orgs = sorted({r["organism"] for r in rows})
    print(f"{len(rows)} نقطه‌ی شکستِ ناحیه‌ای از {len(orgs)} گروهِ ارگانیسم استخراج شد")
    print(f"خروجی: {OUT}")
    print("\nگروه‌ها:")
    for o in orgs:
        n = sum(1 for r in rows if r["organism"] == o)
        print(f"  {n:>4}  {o}")


if __name__ == "__main__":
    main()
