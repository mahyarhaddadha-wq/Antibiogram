# -*- coding: utf-8 -*-
"""Turn the seventeen-slide deck into the twenty-four-slide defence deck:
seven section dividers, a progress bar that fills from the right, a footer
carrying «n / ۲۴ · نام بخش», and a speaker note on every slide."""
import re, html, os, shutil

ORDER = open('order.txt').read().split()
W, Hgt = 12192000, 6858000

# ── the seven sections, their titles and their minute budgets ──────────
AGENDA = [
 ("مسئله و انگیزه",            "چرا خودکار کردن این اندازه‌گیری ارزش دارد", 2.0),
 ("هدف و کارهای پیشین",        "این پژوهش دنبال چیست",                      2.0),
 ("روش پیشنهادی",              "سامانه چگونه کار می‌کند",                   3.5),
 ("نتایج",                     "چقدر دقیق است",                             2.5),
 ("تفسیر بالینی",              "این دقت یعنی چه",                           3.5),
 ("یافته اصلی",                "چرا خطا همان‌جاست",                         2.5),
 ("محدودیت‌ها و جمع‌بندی",      "دستاوردها و قدم بعد",                       3.0),
]
FA = "۰۱۲۳۴۵۶۷۸۹"
fa = lambda n: "".join(FA[int(c)] if c.isdigit() else c for c in str(n))

def mins(x):
    m, s = int(x), round((x - int(x)) * 60)
    return f"{fa(m)} دقیقه" + (f" و {fa(s)} ثانیه" if s else "")

# which agenda item each slide belongs to (0 = none)
SECTION = {1:0, 2:0, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 9:3, 10:3, 11:4, 12:4,
           13:4, 14:5, 15:5, 16:5, 17:6, 18:6, 19:7, 20:7, 21:7, 22:7, 23:0, 24:0}
DIVIDERS = {3:1, 5:2, 8:3, 11:4, 14:5, 17:6, 19:7}

def remaining(sec):          # minutes left from the start of this section
    return sum(a[2] for a in AGENDA[sec-1:])

# ── content of the non-divider slides ──────────────────────────────────
C = {}
C[1] = ["دفاع از پروژه‌ی کارشناسی",
        "طراحی و پیاده‌سازی سامانه هوشمند تحلیل خودکار آزمون آنتی‌بیوگرام",
        "مبتنی بر پردازش تصویر مطابق استاندارد EUCAST",
        "ارائه‌دهنده: مهیار حدادها",
        "کارشناسی مهندسی پزشکی — دانشکده فنی و مهندسی",
        "استاد راهنما: دکتر محمدرضا یزدچی",
        "دانشگاه اصفهان  •  شهریور ۱۴۰۵",
        "دانشگاه", "اصفهان"]

def agenda_body(title, current=None, footer_line=None):
    """The 24 paragraphs of an agenda-shaped slide."""
    out = [title]
    for i, (t, s, m) in enumerate(AGENDA, 1):
        out += [fa(i), t, (s if current is None else
                           (s if i == current else s))]
    out += [footer_line or "فهرست ارائه", None]
    return out

C[2] = agenda_body("فهرست ارائه")
# each agenda subtitle also carries its minute budget on slide 2
for i, (t, s, m) in enumerate(AGENDA, 1):
    C[2][3 * i] = f"{s} — {mins(m)}"

C[4] = ["مسئله و انگیزه",
 "مسئله: قطر ناحیه بدون رشد با خط‌کش و به دست کارشناس اندازه گرفته می‌شود.",
 "چرا اهمیت دارد: همین قطر تعیین می‌کند بیمار کدام آنتی‌بیوتیک را بگیرد.",
 "اشکال روش کنونی: وقت‌گیر است، به تجربه فرد وابسته است، ثبت دیجیتال ندارد.",
 "این پژوهش: همین اندازه‌گیری را از روی یک عکس معمولی خودکار می‌کند.",
 "!", "۱٫۲۷ میلیون مرگ",
 "نسبت‌داده‌شده به مقاومت آنتی‌بیوتیکی در سال ۲۰۱۹",
 None, None]

C[6] = ["هدف و پرسش‌های پژوهش",
 fa(1), "پرسش یکم", "آیا می‌توان قطر هاله را از یک عکس معمولی و بدون تجهیزات ویژه اندازه گرفت؟",
 fa(2), "پرسش دوم", "دقت این اندازه‌گیری در برابر خوانش کارشناس چقدر است؟",
 fa(3), "پرسش سوم", "آیا این دقت برای تصمیم بالینی کافی است؟ اگر نیست، مانع کجاست؟",
 None, None]

C[7] = ["کارهای پیشین و جای این پژوهش",
 "ابزارهای متن‌باز",
 "AntibiogramJ با آستانه‌گذاری کلاسیک، توافق حدود ۸۷ درصد. از نظر فناوری نزدیک‌ترین کار به این پژوهش.",
 "روش‌های یادگیری عمیق",
 "یک برنامه تلفن همراه، توافق ۹۸ درصد. این دسته از نظر دقت عددی از سامانه حاضر جلوتر است.",
 "روش‌های نوری جایگزین",
 "تصویربرداری لکه لیزری، یافتن داروی موثر در کمتر از سه ساعت. به جای بهبود تحلیل، خود تصویربرداری را عوض می‌کند.",
 "جای این پژوهش",
 "یک زنجیره کاملاً قطعی، و اندازه‌گیری علتی که خطای باقی‌مانده را می‌سازد.",
 None, None]

C[9] = ["روش پیشنهادی: چهار مرحله",
 fa(1), "یافتن ظرف",      "مکان و قطر ظرف پتری، و تعیین مقیاس کار",
 fa(2), "یافتن دیسک‌ها",  "ترکیب دو شاخه مستقل؛ قطر شش میلی‌متری استاندارد، مقیاس میلی‌متر را می‌دهد",
 fa(3), "تعیین مرز هاله", "سه روش موازی و ترکیب آن‌ها",
 fa(4), "تفسیر بالینی",   "سنجش قطر با جدول‌های نقطه شکست EUCAST نسخه ۱۶٫۰",
 None, None]

C[10] = ["مرحله سوم: سه روش موازی برای مرز هاله",
 "پروفایل شعاعی",        "پوشش: ۵۷ هاله از ۶۳ — خطای متوسط: ۵٫۴۹ میلی‌متر",
 "آستانه‌گذاری ناحیه‌ای", "پوشش: ۲۶ دیسک — خطای متوسط: ۳٫۶۶ میلی‌متر",
 "رشد کنترل‌شده",        "پوشش: ۱۱ دیسک — خطای متوسط: ۱٫۰۸ میلی‌متر",
 None, None]

C[12] = ["نتایج: یافتن ظرف و یافتن دیسک",
 fa(1), "یافتن ظرف", "۱۱ از ۱۱ عکس. هر ۱۱ عکس از یک آزمایشگاه‌اند؛ این عدد را باید با احتیاط خواند.",
 fa(2), "یافتن دیسک", "F1 برابر ۰٫۹۸۹، با صحت ۱٫۰۰۰ و بازخوانی ۰٫۹۷۸. دو دیسک از دست رفت.",
 fa(3), "تشخیص وجود هاله", "دقت ۰٫۸۲۴. هاله واقعی کم از دست می‌رود؛ گاهی هاله‌ای گزارش می‌شود که نیست.",
 None, None]

C[13] = ["نتایج: دقت قطر هاله و اثر ترکیب",
 "فقط روش پروفایل شعاعی",
 "خطای متوسط: ۵٫۵۰ میلی‌متر", "دقت: ۰٫۷۸۰", "هاله‌های یافته‌شده: ۵۷ از ۶۳",
 "حدود توافق: ۱۶٫۱− تا ۱۵٫۱+ میلی‌متر",
 "ترکیب سه روش",
 "خطای متوسط: ۳٫۸۵ میلی‌متر", "دقت: ۰٫۸۲۴", "هاله‌های یافته‌شده: ۶۰ از ۶۳",
 "حدود توافق: ۱۲٫۳− تا ۱۱٫۱+ میلی‌متر · میانه خطا ۲٫۱۰ · خطا سی درصد کم شد",
 None, None]

C[15] = ["ترجمه خطا به زبان بالینی",
 "۸۳٫۲٪", "توافق دسته‌ای — از ۳۱٬۷۵۹ تصمیم شبیه‌سازی‌شده",
 "۶٫۱۹٪", "خطای بسیار عمده — سقف مرسوم ۱٫۵٪",
 "۹٫۲۴٪", "خطای عمده — سقف مرسوم ۳٫۰٪",
 None, None]

C[16] = ["سقف قابل اثبات: انسان در برابر انسان",
 "کارشناس با کارشناس",
 "خطای متوسط: ۱٫۳۶ میلی‌متر", "توافق دسته‌ای: ۹۵٫۱٪", "خطای بسیار عمده: ۲٫۰۷٪",
 "حتی دو کارشناس انسانی هم سقف ۱٫۵ درصدی را برآورده نمی‌کنند.",
 "سامانه با کارشناس",
 "خطای متوسط: ۳٫۸۵ میلی‌متر", "توافق دسته‌ای: ۸۳٫۲٪", "خطای بسیار عمده: ۶٫۱۹٪",
 "سقف واقعی حدود ۹۲ تا ۹۶ درصد است؛ فاصله سامانه ده واحد است، نه هجده.",
 None, None]

C[18] = ["یافته اصلی: سقفی که چهار روش مستقل به آن رسیدند",
 "اندازه‌گیری: روی نیمی از دیسک‌ها، روشنایی داخل هاله تنها ۰٫۳ انحراف معیار فضایی محیط کشت با آن فرق دارد.",
 "کنترل یکم: مرجع داخل هاله در صفر درصد موارد بیرون هاله نیفتاده است.",
 "کنترل دوم: با دادن محل درست مرز به الگوریتم، جدایش به ۰٫۲۹ می‌رسد.",
 "دو راه استاندارد رد شد: تغییر فضای نوری خطا را به ۷٫۰۰ برد و میانگین‌گیری فضایی ۲٫۶۳ میلی‌متر تاری افزود.",
 "!", "پیام کلیدی",
 "این اطلاعات به‌کلی از تصویر حذف نشده، اما هیچ‌یک از روش‌های این پروژه نتوانست در این سطح کنتراست به آن دست یابد.",
 None, None]

C[20] = ["محدودیت‌ها",
 fa(1), "حجم و تنوع داده", "یازده عکس و ۹۳ دیسک، همه از یک آزمایشگاه با یک دوربین. توصیف اولیه از دقت، نه اعتبارسنجی بالینی.",
 fa(2), "پروتکل تصویربرداری", "عکس‌ها با تلفن همراه و در نور اتاق گرفته شده‌اند؛ استاندارد نور بازتابی و زمینه تیره را توصیه می‌کند.",
 fa(3), "مرجع و دامنه", "عدد دو میلی‌متری کارشناس یک اظهار شفاهی است. نام دارو خوانده نمی‌شود. زمان اجرا یک تا سه دقیقه است.",
 None, None]

C[21] = ["نتیجه‌گیری و دستاوردها",
 "۰۱", "زنجیره‌ای کامل و قطعی از عکس خام تا دسته بالینی. نتیجه روی سه ماشین مستقل بازتولید شد، در یکی بیت به بیت.",
 "۰۲", "مرز هاله با سه روش موازی و ترکیب بر پایه نقاط قوت اندازه‌گیری‌شده. این ترکیب خطا را سی درصد کاهش داد.",
 "۰۳", "هر تصمیم طراحی با اندازه‌گیری سنجیده شد، از جمله معیاری که در اعتبارسنجی متقاطع رد شد و کنار گذاشته شد.",
 "یافتن دیسک حل شده است. تعیین مرز هاله تا حدی حل شده، و علت باقی‌مانده‌اش اندازه‌گیری شده است، نه حدس زده.",
 None, None]

C[22] = ["کار آینده",
 "کوتاه‌مدت", "افزودن مدل سیگموئید گذار شعاعی. سنجش آفلاین روی ۵۷ دیسک مشترک: خطای ۲٫۷۹ و میانه ۱٫۰۳ میلی‌متر.",
 "میان‌مدت", "چهار شرط تصویربرداری: نور از پشت ظرف، کارت خاکستری مرجع، عکس در چند زمان، و مربع مرجع ۱۰ میلی‌متری.",
 "بلندمدت", "مجموعه مرجع بزرگ‌تر با چند آزمایشگاه و چند دوربین، و دور دوم اندازه‌گیری کارشناسی.",
 None, None]

C[23] = ["♥", "سپاسگزاری",
 "از استاد راهنما، جناب آقای دکتر محمدرضا یزدچی",
 "از کارشناسان آزمایشگاه، برای اندازه‌گیری دستی ۶۴ هاله",
 "از هیئت داوران، برای وقتی که گذاشتند",
 "از خانواده‌ام", None, None]

C[24] = ["سپاس از توجه شما", "پرسش و پاسخ", "مهیار حدادها  •  ۴۰۱۲۰۱۳۰۵۳"]

# dividers
for pos, sec in DIVIDERS.items():
    C[pos] = agenda_body(f"بخش {fa(sec)} از ۷ — حدود {mins(remaining(sec))} تا پایان",
                         current=sec, footer_line=AGENDA[sec-1][0])

# ── helpers ────────────────────────────────────────────────────────────
P_RE = re.compile(r"<a:p>(?:(?!</a:p>).)*</a:p>|<a:p/>", re.S)
R_RE = re.compile(r"<a:r>(?:(?!</a:r>).)*</a:r>", re.S)

def set_text(par, text):
    runs = R_RE.findall(par)
    if not runs: return par
    new = re.sub(r"<a:t>.*?</a:t>", "<a:t>" + html.escape(text, quote=False) + "</a:t>",
                 runs[0], count=1, flags=re.S)
    out = par.replace(runs[0], new, 1)
    for extra in runs[1:]: out = out.replace(extra, "", 1)
    return out

def bar(pct):
    """Two rectangles: the track, and the filled part hugging the right edge."""
    h, y = 55000, Hgt - 55000
    w = int(W * pct)
    def rect(x, cx, colour, sid, name):
        return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{h}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
                f'<a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>'
                f'<a:ln><a:noFill/></a:ln></p:spPr>'
                f'<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>')
    return rect(0, W, "EDE9E5", 9001, "progress track") + rect(W - w, w, "8C1515", 9002, "progress fill")

GREY_T, GREY_S, GREY_C = "C3C0BC", "CFCCC8", "DCD8D4"

# ── the pass ───────────────────────────────────────────────────────────
for pos, fn in enumerate(ORDER, 1):
    path = f"u/ppt/slides/{fn}"
    x = open(path, encoding="utf8").read()
    pars = P_RE.findall(x)
    want = C[pos]
    assert len(pars) == len(want), f"slide {pos} ({fn}): {len(pars)} paragraphs, {len(want)} texts"

    sec = SECTION[pos]
    for k, (par, text) in enumerate(zip(pars, want)):
        if text is None:                       # footer strip / slide number
            text = (AGENDA[sec-1][0] if sec and k == len(want) - 2 else
                    "" if k == len(want) - 2 else f"{fa(pos)} / ۲۴")
            if k == len(want) - 2 and pos in DIVIDERS: text = AGENDA[DIVIDERS[pos]-1][0]
        x = x.replace(par, set_text(par, text), 1)

    # a divider dims every section but the current one
    if pos in DIVIDERS:
        cur = DIVIDERS[pos]
        sps = re.findall(r"<p:sp>.*?</p:sp>", x, re.S)
        for i, sp in enumerate(sps):
            item = None
            for n in range(7):
                if i in (2 + n*4, 3 + n*4, 4 + n*4, 5 + n*4): item = n + 1
            if item and item != cur:
                new = sp
                new = re.sub(r'<a:srgbClr val="(?:8C1515|E98300)"/>', f'<a:srgbClr val="{GREY_C}"/>', new)
                new = re.sub(r'<a:srgbClr val="2E2D29"/>', f'<a:srgbClr val="{GREY_T}"/>', new)
                new = re.sub(r'<a:srgbClr val="716E69"/>', f'<a:srgbClr val="{GREY_S}"/>', new)
                x = x.replace(sp, new, 1)

    # the progress bar, on every slide but the two dark covers
    if pos not in (1, 24):
        x = x.replace("</p:spTree>", bar(pos / 24) + "</p:spTree>")

    # the thesis is set in B Nazanin, and so is the deck
    x = x.replace('<a:cs typeface="IRTitr"', '<a:cs typeface="B Titr"')
    x = x.replace('<a:cs typeface="IRNazanin"', '<a:cs typeface="B Nazanin"')
    open(path, "w", encoding="utf8").write(x)
    print(f"  {pos:>2} {fn:12} ok")

t = open("u/ppt/theme/theme1.xml", encoding="utf8").read()
t = t.replace('<a:cs typeface="IRTitr"/>', '<a:cs typeface="B Titr"/>')
t = t.replace('<a:cs typeface="IRNazanin"/>', '<a:cs typeface="B Nazanin"/>')
open("u/ppt/theme/theme1.xml", "w", encoding="utf8").write(t)
print("fonts set to B Nazanin / B Titr")
