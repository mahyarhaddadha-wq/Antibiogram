# -*- coding: utf-8 -*-
"""Three finishing passes: the full approved title, Latin digits inside the
chart, and a Persian speaker note with a time budget on every slide."""
import re, html, shutil, os

SLIDE_W = 12192000

# ── 1. the title slide carries the full approved title ────────────────────
p = "unpacked/ppt/slides/slide1.xml"
x = open(p, encoding="utf8").read()
par = re.search(r"<a:p>(?:(?!</a:p>).)*طراحی(?:(?!</a:p>).)*</a:p>", x, re.S).group(0)
line1 = par.replace("sz=\"4000\"", "sz=\"3200\"")
line2 = (par.replace("sz=\"4000\"", "sz=\"1800\"")
            .replace("<a:t>طراحی و پیاده‌سازی سامانه هوشمند تحلیل خودکار آزمون آنتی‌بیوگرام</a:t>",
                     "<a:t>مبتنی بر پردازش تصویر و یادگیری ماشین مطابق استاندارد EUCAST</a:t>"))
assert "EUCAST" in line2
x = x.replace(par, line1 + line2, 1)
open(p, "w", encoding="utf8").write(x)
print("  title slide: two-line title set")

# ── 2. the chart reads in one numeral system ──────────────────────────────
p = "unpacked/ppt/charts/chart1.xml"
c = open(p, encoding="utf8").read()
for fa, en in zip(["۳٫۸۵", "۲٫۸۹", "۱٫۹۳", "۱٫۳۵", "۰٫۹۶"],
                  ["3.85", "2.89", "1.93", "1.35", "0.96"]):
    c = c.replace(f"<c:v>{fa}</c:v>", f"<c:v>{en}</c:v>")
open(p, "w", encoding="utf8").write(c)
print("  chart: category labels set in Latin digits, matching the values")

# ── 3. speaker notes, with the twenty minutes budgeted ────────────────────
NOTES = {
 "slide1": "«۳۰ ثانیه» با سلام و خوش‌آمد شروع کنید. عنوان، رشته و نام استاد راهنما را شمرده بگویید. عجله نکنید؛ داوران در همین سی ثانیه ذهنشان را تنظیم می‌کنند.",
 "slide2": "«۳۰ ثانیه» فقط هفت سرفصل را بخوانید. توضیح ندهید. بگویید ارائه حدود بیست دقیقه است و پرسش‌ها در پایان پاسخ داده می‌شود.",
 "slide3": "«۱ دقیقه و ۳۰ ثانیه» با تصویر ذهنی شروع کنید: کارشناس، خط‌کش و ظرف پتری. بعد بگویید نتیجه همین اندازه‌گیری، داروی بیمار را تعیین می‌کند. عدد ۱٫۲۷ میلیون را آرام بگویید و مکث کنید.",
 "slide4": "«۱ دقیقه» سه پرسش را بخوانید و تاکید کنید که پرسش سوم مهم‌ترین است. بگویید پاسخ پرسش سوم در این ارائه صادقانه داده می‌شود، چه مثبت باشد چه منفی.",
 "slide5": "«۱ دقیقه» صریح بگویید که دسته یادگیری عمیق از این سامانه دقیق‌تر است. پنهان کردن این نکته، اولین پرسش داور می‌شود. بعد بگویید سهم این کار چیز دیگری است.",
 "slide6": "«۲ دقیقه» چهار مرحله را به ترتیب و هر کدام در سی ثانیه بگویید. روی مرحله دوم تاکید کنید: قطر شش میلی‌متری دیسک، تنها مرجع مطلق در کل تصویر است و مقیاس از همان‌جا می‌آید.",
 "slide17": "«۱ دقیقه و ۳۰ ثانیه» پیام این اسلاید یک جمله است: هیچ روشی هم پوشش خوب دارد و هم دقت خوب. سه عدد پوشش و سه عدد خطا را کنار هم بگذارید و بگویید ترکیب دقیقاً برای همین ساخته شد.",
 "slide15": "«۱ دقیقه» اینجا خبر خوب است. سریع بگویید و نمانید. جمله احتیاط درباره یک آزمایشگاه بودن را حتماً بگویید؛ داور آن را می‌پرسد و بهتر است خودتان گفته باشید.",
 "slide8": "«۱ دقیقه و ۳۰ ثانیه» دو ستون را مقایسه کنید. نکته اصلی: ترکیب روی هر سه معیار بهتر یا برابر است. عدد سی درصد کاهش خطا را برجسته بگویید.",
 "slide7": "«۲ دقیقه» اینجا صادق‌ترین بخش ارائه است. بگویید سامانه به سقف پذیرش بالینی نمی‌رسد. بعد به نمودار بروید و بگویید برای رسیدن به آن سقف، خطا باید به حدود یک میلی‌متر برسد. این عدد، هدف مهندسی است.",
 "slide16": "«۱ دقیقه و ۳۰ ثانیه» این اسلاید تفسیر اسلاید قبل را عوض می‌کند. بگویید حتی دو کارشناس انسانی هم سقف یک و نیم درصد را برآورده نمی‌کنند. پس سقف واقعی صد درصد نیست و فاصله سامانه ده واحد است، نه هجده واحد.",
 "slide9": "«۲ دقیقه و ۳۰ ثانیه» مهم‌ترین اسلاید ارائه است. آرام بگویید. عدد سه دهم را توضیح دهید، بعد دو کنترل را بگویید تا نشان دهید عدد مصنوع روش نیست، و در پایان جمله کادر را مستقیم بخوانید.",
 "slide10": "«۱ دقیقه» محدودیت‌ها را بدون دفاع و بدون عذرخواهی بگویید. لحن باید خونسرد باشد. این بخش اعتماد داور را می‌سازد.",
 "slide11": "«۱ دقیقه» سه دستاورد را بشمارید. جمله جمع‌بندی پایین اسلاید را کلمه به کلمه بخوانید؛ همان یک جمله است که در ذهن داور می‌ماند.",
 "slide12": "«۱ دقیقه» تاکید کنید که پیشنهاد کوتاه‌مدت از پیش سنجیده شده و عدد دارد. سه شرط تصویربرداری را بگویید و اضافه کنید که هیچ‌کدام تجهیزات گران نمی‌خواهند.",
 "slide13": "«۲۰ ثانیه» کوتاه و صمیمانه. اسم‌ها را درست تلفظ کنید.",
 "slide14": "این اسلاید تا پایان جلسه روی پرده می‌ماند. برای پرسش‌ها آماده باشید: چرا یادگیری ماشین نه، چرا فقط یازده عکس، و چرا خطای بسیار عمده بالاست. پاسخ هر سه در اسلایدهای ۵، ۱۳ و ۱۱ آمده است.",
}

CT = "unpacked/[Content_Types].xml"
ct = open(CT, encoding="utf8").read()

for slide, text in NOTES.items():
    ns_path = f"unpacked/ppt/notesSlides/notes{slide.capitalize()}.xml"
    rels = f"unpacked/ppt/slides/_rels/{slide}.xml.rels"
    r = open(rels, encoding="utf8").read()
    m = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', r)
    if m:
        ns_path = "unpacked/ppt/notesSlides/" + m.group(1)
    else:
        # A duplicated slide has no notes part yet; give it one.
        n = slide.replace("slide", "")
        ns_name = f"notesSlide{n}.xml"
        ns_path = "unpacked/ppt/notesSlides/" + ns_name
        shutil.copy("unpacked/ppt/notesSlides/notesSlide1.xml", ns_path)
        open(f"unpacked/ppt/notesSlides/_rels/{ns_name}.rels", "w", encoding="utf8").write(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships/slide" Target="../slides/{slide}.xml"/></Relationships>')
        rid = "rId%d" % (max(int(i) for i in re.findall(r'Id="rId(\d+)"', r)) + 1)
        r = r.replace("</Relationships>",
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/'
            f'2006/relationships/notesSlide" Target="../notesSlides/{ns_name}"/></Relationships>')
        open(rels, "w", encoding="utf8").write(r)
        ct = ct.replace("</Types>",
            f'<Override PartName="/ppt/notesSlides/{ns_name}" ContentType="application/vnd.'
            f'openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>')

    ns = open(ns_path, encoding="utf8").read()
    body = re.search(r'(<p:ph type="body" idx="1"/>.*?)<a:t>.*?</a:t>', ns, re.S)
    ns = (ns[:body.start()] + body.group(1) + "<a:t>" + html.escape(text, quote=False) + "</a:t>"
          + ns[body.end():])
    ns = ns.replace('<a:p><a:r><a:rPr lang="en-US" dirty="0"/>',
                    '<a:p><a:pPr rtl="1" algn="r"/><a:r>'
                    '<a:rPr lang="fa-IR" dirty="0"><a:cs typeface="IRNazanin"/></a:rPr>')
    open(ns_path, "w", encoding="utf8").write(ns)

open(CT, "w", encoding="utf8").write(ct)
print(f"  notes: {len(NOTES)} slides carry a Persian note with its time budget")
