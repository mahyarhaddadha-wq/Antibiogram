# -*- coding: utf-8 -*-
"""Put a Persian speaker note, with its minute budget, on all 24 slides."""
import re, html, shutil, os

ORDER = open('order.txt').read().split()
N = {
1:"«۲۰ ثانیه» سلام و خوش‌آمد. عنوان، رشته و نام استاد راهنما را شمرده بگویید. عجله نکنید.",
2:"«۴۰ ثانیه» هفت سرفصل را بخوانید و بگویید ارائه بیست دقیقه است و پرسش‌ها در پایان. توضیح ندهید.",
3:"«۵ ثانیه» فقط عنوان بخش را بگویید و رد شوید. این اسلاید برای مخاطب است، نه برای شما.",
4:"«۲ دقیقه» با تصویر ذهنی شروع کنید: کارشناس، خط‌کش، ظرف پتری. بعد بگویید همین قطر داروی بیمار را تعیین می‌کند. عدد ۱٫۲۷ میلیون را آرام بگویید و مکث کنید.",
5:"«۵ ثانیه» رد شوید.",
6:"«۱ دقیقه» سه پرسش را بخوانید و تاکید کنید پرسش سوم مهم‌ترین است و پاسخش صادقانه داده خواهد شد، چه مثبت چه منفی.",
7:"«۱ دقیقه» صریح بگویید دسته یادگیری عمیق دقیق‌تر از این سامانه است. پنهان کردنش اولین پرسش داور می‌شود. بعد بگویید سهم این کار چیز دیگری است.",
8:"«۵ ثانیه» رد شوید.",
9:"«۲ دقیقه» چهار مرحله را هر کدام در سی ثانیه. روی مرحله دوم تاکید کنید: قطر شش میلی‌متری دیسک تنها مرجع مطلق در کل تصویر است. در پایان بگویید هیچ پارامتری بر حسب پیکسل مطلق نیست.",
10:"«۱ دقیقه و ۳۰ ثانیه» سه عدد پوشش و سه عدد خطا را کنار هم بگذارید. پیام یک جمله است: هیچ روشی هم پوشش خوب دارد و هم دقت خوب؛ ترکیب برای همین ساخته شد.",
11:"«۵ ثانیه» رد شوید.",
12:"«۱ دقیقه» خبر خوب است، سریع بگویید. جمله احتیاط درباره یک آزمایشگاه بودن را حتماً بگویید؛ داور آن را می‌پرسد و بهتر است خودتان گفته باشید.",
13:"«۱ دقیقه و ۳۰ ثانیه» دو ستون را مقایسه کنید. نکته اصلی: ترکیب روی هر سه معیار بهتر یا برابر است، و خطا سی درصد کم شد. اضافه کنید که سوگیری تقریباً صفر است.",
14:"«۵ ثانیه» رد شوید.",
15:"«۲ دقیقه» صادق‌ترین بخش ارائه. اول بگویید طبقه‌بندی به قطر و گونه و نام دارو نیاز دارد و مرجع فقط قطر را داشت، پس پرسش را عوض کردیم. سه عدد را بگویید و بعد صریح: سامانه به آستانه پذیرش بالینی نمی‌رسد. مکث. بعد به نمودار بروید: هدف مهندسی خطای حدود یک میلی‌متر است.",
16:"«۱ دقیقه و ۳۰ ثانیه» این اسلاید تفسیر اسلاید قبل را عوض می‌کند. بگویید حتی دو کارشناس انسانی هم سقف یک و نیم درصد را برآورده نمی‌کنند، پس سقف واقعی صد درصد نیست. در پایان صادقانه اضافه کنید که با کسر نویز مرجع خطا فقط به ۳٫۷ می‌رسد؛ این نویز بهانه نیست.",
17:"«۵ ثانیه» رد شوید.",
18:"«۲ دقیقه و ۳۰ ثانیه» مهم‌ترین اسلاید. آرام بگویید. اول الگوی تکرارشونده، بعد عدد سه دهم، بعد دو کنترل تا نشان دهید عدد مصنوع روش نیست، بعد دو راه استانداردی که رد شد. در پایان جمله کادر را کلمه به کلمه بخوانید.",
19:"«۵ ثانیه» رد شوید.",
20:"«۱ دقیقه» بدون دفاع و بدون عذرخواهی. لحن خونسرد. این بخش اعتماد داور را می‌سازد.",
21:"«۱ دقیقه» سه دستاورد را بشمارید. روی سومی تاکید کنید، چون همان چیزی است که چکیده هم بر آن ایستاده. جمله جمع‌بندی پایین را کلمه به کلمه بخوانید.",
22:"«۱ دقیقه» تاکید کنید پیشنهاد کوتاه‌مدت از پیش سنجیده شده و عدد دارد. چهار شرط تصویربرداری را بگویید و اضافه کنید هیچ‌کدام تجهیزات گران نمی‌خواهند.",
23:"«۲۰ ثانیه» کوتاه و صمیمانه. اسم‌ها را درست تلفظ کنید.",
24:"تا پایان جلسه روی پرده می‌ماند. پرسش‌های محتمل و شماره اسلاید پاسخشان: چرا بدون یادگیری ماشین (۷ و ۱۸)؛ چرا فقط یازده عکس (۲۰)؛ چرا خطای بسیار عمده بالاست (۱۵ و ۱۶)؛ چرا مقیاس از قطر دیسک (۹ و ۲۲)؛ اگر مدل سیگموئید بهتر است چرا در نتایج نیست (۲۲).",
}

CT = "u/[Content_Types].xml"; ct = open(CT, encoding="utf8").read()
made = 0
for pos, fn in enumerate(ORDER, 1):
    stem = fn[:-4]
    rp = f"u/ppt/slides/_rels/{fn}.rels"; r = open(rp, encoding="utf8").read()
    m = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', r)
    if m:
        ns = "u/ppt/notesSlides/" + m.group(1)
    else:
        name = f"notes{stem.capitalize()}.xml"
        ns = "u/ppt/notesSlides/" + name
        shutil.copy("u/ppt/notesSlides/notesSlide1.xml", ns)
        open(f"u/ppt/notesSlides/_rels/{name}.rels", "w", encoding="utf8").write(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships/slide" Target="../slides/{fn}"/></Relationships>')
        rid = "rId%d" % (max(int(i) for i in re.findall(r'Id="rId(\d+)"', r)) + 1)
        open(rp, "w", encoding="utf8").write(r.replace("</Relationships>",
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/'
            f'2006/relationships/notesSlide" Target="../notesSlides/{name}"/></Relationships>'))
        ct = ct.replace("</Types>", f'<Override PartName="/ppt/notesSlides/{name}" '
            'ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>')
        made += 1
    x = open(ns, encoding="utf8").read()
    b = re.search(r'(<p:ph type="body" idx="1"/>.*?)<a:t>.*?</a:t>', x, re.S)
    x = x[:b.start()] + b.group(1) + "<a:t>" + html.escape(N[pos], quote=False) + "</a:t>" + x[b.end():]
    x = re.sub(r'<a:p><a:pPr rtl="1" algn="r"/><a:r><a:rPr lang="[^"]*" dirty="0"><a:cs typeface="[^"]*"/></a:rPr>',
               '<a:p><a:pPr rtl="1" algn="r"/><a:r><a:rPr lang="fa-IR" dirty="0"><a:cs typeface="B Nazanin"/></a:rPr>', x)
    x = x.replace('<a:p><a:r><a:rPr lang="en-US" dirty="0"/>',
                  '<a:p><a:pPr rtl="1" algn="r"/><a:r><a:rPr lang="fa-IR" dirty="0"><a:cs typeface="B Nazanin"/></a:rPr>')
    open(ns, "w", encoding="utf8").write(x)
open(CT, "w", encoding="utf8").write(ct)
print(f"notes on 24 slides ({made} new notes parts)")
