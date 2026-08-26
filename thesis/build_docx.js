#!/usr/bin/env node
/**
 * Build the final Persian thesis .docx from the markdown chapters.
 *
 * Layout follows thesis_template.docx: A4, RTL, IRNazanin, 1985/1701 twip
 * margins, chapter-scoped figure and table numbering, IEEE references.
 *
 *   node thesis/build_docx.js
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");

const THESIS = __dirname;
const REPO = path.dirname(THESIS);
const FIGS = path.join(THESIS, "figures");
const GALLERY = path.join(REPO, "pipeline_module_gallery", "gt_06");

// ─── Template constants ────────────────────────────────────────────────
const FONT = "IRNazanin";
const FONT_EN = "Times New Roman";
const CONTENT_W = 11906 - 1985 * 2; // usable width in DXA

const SZ = { body: 26, h1: 30, h2: 28, h3: 26, cap: 24, tbl: 22, small: 20 };

const PAGE = {
  size: { width: 11906, height: 16838 },
  margin: { top: 1701, right: 1985, bottom: 1701, left: 1985 },
};

// ─── Inline formatting ─────────────────────────────────────────────────
// Splits on **bold**, *italic*, `code` and $math$, keeping RTL on each run.
function runs(text, opt = {}) {
  const size = opt.size || SZ.body;
  const base = { font: FONT, size, rightToLeft: true, ...(opt.run || {}) };
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\$[^$]+\$)/g;
  let last = 0, m;
  const push = (t, extra) => {
    if (!t) return;
    // Latin/digit-only fragments render with the Latin face.
    const latin = /^[\x20-\x7E\u2212\u2018\u2019\u201C\u201D]+$/.test(t);
    out.push(new D.TextRun({
      ...base, text: t, ...extra,
      ...(latin ? { font: FONT_EN, rightToLeft: false } : {}),
    }));
  };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: SZ.small, rightToLeft: false });
    else if (tok.startsWith("$")) push(tok.slice(1, -1), { font: FONT_EN, italics: true, rightToLeft: false });
    else push(tok.slice(1, -1), { italics: true });
    last = re.lastIndex;
  }
  push(text.slice(last));
  return out.length ? out : [new D.TextRun({ ...base, text: "" })];
}

const para = (text, o = {}) => new D.Paragraph({
  children: runs(text, o),
  bidirectional: true,
  alignment: o.alignment || D.AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, line: o.line ?? 300 },
  indent: o.indent,
  ...(o.border ? { border: o.border } : {}),
  ...(o.heading ? { heading: o.heading } : {}),
  ...(o.pageBreakBefore ? { pageBreakBefore: true } : {}),
});

const blank = (after = 120) => new D.Paragraph({ text: "", spacing: { after } });

// ─── Headings ──────────────────────────────────────────────────────────
function heading(text, level) {
  const cfg = {
    1: { size: SZ.h1, heading: D.HeadingLevel.HEADING_1, before: 0, after: 260, align: D.AlignmentType.CENTER },
    2: { size: SZ.h2, heading: D.HeadingLevel.HEADING_2, before: 280, after: 140, align: D.AlignmentType.RIGHT },
    3: { size: SZ.h3, heading: D.HeadingLevel.HEADING_3, before: 200, after: 110, align: D.AlignmentType.RIGHT },
  }[level];
  return new D.Paragraph({
    children: [new D.TextRun({ font: FONT, size: cfg.size, bold: true, rightToLeft: true, text })],
    heading: cfg.heading,
    bidirectional: true,
    alignment: cfg.align,
    spacing: { before: cfg.before, after: cfg.after },
    ...(level === 1 ? { pageBreakBefore: true } : {}),
  });
}

// ─── Captions ──────────────────────────────────────────────────────────
const caption = (text) => new D.Paragraph({
  children: runs(text, { size: SZ.cap, run: { bold: true } }),
  bidirectional: true,
  alignment: D.AlignmentType.CENTER,
  spacing: { before: 80, after: 220 },
});

// ─── Images ────────────────────────────────────────────────────────────
function image(file, maxWidthPt = 400) {
  const buf = fs.readFileSync(file);
  // PNG/JPEG intrinsic size, to preserve aspect ratio.
  let w, h;
  if (buf.slice(1, 4).toString() === "PNG") {
    w = buf.readUInt32BE(16); h = buf.readUInt32BE(20);
  } else {
    let i = 2;
    while (i < buf.length) {
      if (buf[i] !== 0xff) { i++; continue; }
      const mk = buf[i + 1];
      if (mk >= 0xc0 && mk <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(mk)) {
        h = buf.readUInt16BE(i + 5); w = buf.readUInt16BE(i + 7); break;
      }
      i += 2 + buf.readUInt16BE(i + 2);
    }
  }
  const scale = Math.min(1, maxWidthPt / w);
  return new D.Paragraph({
    children: [new D.ImageRun({
      type: path.extname(file).toLowerCase() === ".png" ? "png" : "jpg",
      data: buf,
      transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
    })],
    alignment: D.AlignmentType.CENTER,
    spacing: { before: 200, after: 40 },
  });
}

// ─── Tables ────────────────────────────────────────────────────────────
const SHADE = "EDF2F9";

function table(rows) {
  const nCols = Math.max(...rows.map((r) => r.length));
  const colW = Math.floor(CONTENT_W / nCols);
  const widths = Array(nCols).fill(colW);
  widths[nCols - 1] = CONTENT_W - colW * (nCols - 1);

  const mk = (cells, isHead) => new D.TableRow({
    tableHeader: isHead,
    cantSplit: true,
    children: Array.from({ length: nCols }, (_, i) => new D.TableCell({
      width: { size: widths[i], type: D.WidthType.DXA },
      shading: isHead ? { type: D.ShadingType.CLEAR, fill: SHADE } : undefined,
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      children: [new D.Paragraph({
        children: runs(cells[i] ?? "", { size: SZ.tbl, run: { bold: !!isHead } }),
        bidirectional: true,
        alignment: D.AlignmentType.CENTER,
        spacing: { after: 0, line: 260 },
      })],
    })),
  });

  return new D.Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: D.WidthType.DXA },
    visuallyRightToLeft: true,
    rows: [mk(rows[0], true), ...rows.slice(1).map((r) => mk(r, false))],
  });
}

// ─── Figure manifest ───────────────────────────────────────────────────
const galleryFiles = fs.readdirSync(GALLERY).filter((f) => f.endsWith(".png")).sort();
const FIGURES = { "۳-۱": path.join(FIGS, "fig_3_1_architecture.png") };
galleryFiles.forEach((f, i) => { FIGURES[`۴-${toFa(i + 1)}`] = path.join(GALLERY, f); });
[["۵-۱", "fig_5_1_bland_altman.png"], ["۵-۲", "fig_5_2_system_vs_expert.png"],
 ["۵-۳", "fig_5_3_best_case_gt03.png"], ["۵-۴", "fig_5_4_worst_case_gt04.png"]]
  .forEach(([k, f]) => { FIGURES[k] = path.join(FIGS, f); });

function toFa(n) { return String(n).replace(/\d/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]); }

// ─── Markdown → docx blocks ────────────────────────────────────────────
function convert(md, opts = {}) {
  const out = [];
  const lines = md.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (!t) { i++; continue; }
    if (/^---+$/.test(t)) { i++; continue; }              // horizontal rule
    if (/^<\/?div/.test(t)) { i++; continue; }            // html wrappers
    if (/^>\s*\*\*راهنمای استفاده/.test(t)) {             // editor-only note
      while (i < lines.length && lines[i].trim()) i++;
      continue;
    }

    // Headings
    let m;
    if ((m = t.match(/^(#{1,4})\s+(.*)$/))) {
      const depth = m[1].length;
      const text = m[2].replace(/\*\*/g, "").trim();
      // Sections the builder renders natively (title pages, generated TOC)
      // are dropped here so they are not duplicated from the markdown.
      if ((opts.skip || []).some((k) => text.includes(k))) {
        i++;
        while (i < lines.length && !new RegExp(`^#{1,${depth}} `).test(lines[i])) i++;
        continue;
      }
      if (depth === 1) out.push(heading(text, 1));
      else if (depth === 2) out.push(heading(text, opts.demote ? 3 : 2));
      else out.push(heading(text, 3));
      i++; continue;
    }

    // Fenced code
    if (t.startsWith("```")) {
      const body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) body.push(lines[i]), i++;
      i++;
      // The ASCII pipeline sketch is superseded by figure 3-1.
      if (body.join("\n").includes("تصویر ورودی (ممکن است")) continue;
      body.forEach((b) => out.push(new D.Paragraph({
        children: [new D.TextRun({ font: "Consolas", size: SZ.small, text: b || " ", rightToLeft: false })],
        alignment: D.AlignmentType.LEFT,
        spacing: { after: 0, line: 240 },
        bidirectional: false,
      })));
      out.push(blank());
      continue;
    }

    // Tables
    if (t.startsWith("|") && (lines[i + 1] || "").trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        if (!/^[\s:\-|]+$/.test(cells.join(""))) rows.push(cells);
        i++;
      }
      if (rows.length) { out.push(table(rows)); out.push(blank(200)); }
      continue;
    }

    // Figure caption lines → image + caption
    if ((m = t.match(/^\*{1,2}(?:شکلِ?|شکل)\s*([۰-۹]+-[۰-۹]+)\s*[:：](.*)$/))) {
      const num = m[1];
      const capText = ("شکل " + num + ": " + m[2]).replace(/\*+$/, "").replace(/\s*\([^)]*\.png\)\s*$/, "").trim();
      const f = FIGURES[num];
      if (f && fs.existsSync(f)) out.push(image(f, num.startsWith("۴") ? 300 : 400));
      out.push(caption(capText));
      i++; continue;
    }

    // Blockquote
    if (t.startsWith(">")) {
      const body = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        body.push(lines[i].trim().replace(/^>\s?/, "")); i++;
      }
      body.filter((b) => b.trim()).forEach((b) => out.push(para(b, {
        indent: { right: 340 },
        border: { right: { style: D.BorderStyle.SINGLE, size: 12, color: "2A78D6", space: 10 } },
        after: 90,
      })));
      out.push(blank(140));
      continue;
    }

    // Lists
    if (/^[-*]\s+/.test(t) || /^\d+[.)]\s+/.test(t) || /^[۰-۹]+[.)]\s+/.test(t)) {
      const bullet = /^[-*]\s+/.test(t);
      out.push(para(t.replace(/^[-*]\s+/, "• ").replace(/^([۰-۹\d]+[.)])\s+/, "$1 "), {
        indent: { right: bullet ? 300 : 300, hanging: 180 },
        after: 70,
      }));
      i++; continue;
    }

    // Plain paragraph (join soft-wrapped lines)
    const buf = [t];
    i++;
    while (i < lines.length) {
      const n = lines[i].trim();
      if (!n || /^[#>|`-]/.test(n) || /^\d+[.)]\s/.test(n) || /^[۰-۹]+[.)]\s/.test(n)
          || /^\*{1,2}شکل/.test(n) || /^<\/?div/.test(n)) break;
      buf.push(n); i++;
    }
    out.push(para(buf.join(" ")));
  }
  return out;
}

// ─── Document assembly ─────────────────────────────────────────────────
const read = (f) => fs.readFileSync(path.join(THESIS, f), "utf8");

const centered = (text, size, bold = true, after = 160, en = false) => new D.Paragraph({
  children: [new D.TextRun({
    font: en ? FONT_EN : FONT, size, bold, text, rightToLeft: !en,
  })],
  alignment: D.AlignmentType.CENTER,
  bidirectional: !en,
  spacing: { after },
});

// ---- Title page (Persian) ----
const titlePage = [
  blank(600),
  centered("دانشگاه اصفهان", 30),
  centered("دانشکده‌ی فنی و مهندسی", 26),
  centered("گروه مهندسی پزشکی", 24, true, 900),
  centered("پروژه‌ی کارشناسی رشته‌ی مهندسی پزشکی", 32, true, 700),
  centered("طراحی و پیاده‌سازی سامانه هوشمند تحلیل خودکار آزمون آنتی‌بیوگرام", 30, true, 90),
  centered("مبتنی بر پردازش تصویر و یادگیری ماشین مطابق استاندارد EUCAST", 30, true, 900),
  centered("استاد راهنما:", 28, true, 90),
  centered("دکتر محمدرضا یزدچی", 28, true, 600),
  centered("دانشجو:", 28, true, 90),
  centered("مهیار حدادها", 28, true, 60),
  centered("۴۰۱۲۰۱۳۰۵۳", 26, true, 900),
  centered("شهریور ۱۴۰۵", 26),
  new D.Paragraph({ children: [new D.PageBreak()] }),
];

// ---- Front matter, chapters, back matter ----
// ONLY=<n> builds a reduced document, for isolating a rendering fault.
const ONLY = process.env.ONLY ? Number(process.env.ONLY) : 0;
const pieces = [
  ...titlePage,
  ...convert(read("front_matter.md"), {
    demote: false,
    skip: ["صفحه‌ی عنوان (فارسی)", "فهرست مطالب", "English Title Page"],
  }),
  heading("فهرست مطالب", 1),
  new D.TableOfContents("فهرست مطالب", { hyperlink: true, headingStyleRange: "1-3", rightTabStop: CONTENT_W }),
  ...convert(read("chapter_01.md")),
  ...convert(read("chapter_02.md")),
  ...convert(read("chapter_03.md")),
  ...convert(read("chapter_04.md")),
  ...convert(read("chapter_05.md")),
  ...convert(read("chapter_06.md")),
  heading("منابع و مآخذ", 1),
  ...convert(refsSection()),
  heading("پیوست ب — داده‌ی کامل ارزیابی", 1),
  ...convert(read("appendix_b_evaluation_data.md").replace(/^#\s+.*$/m, ""), { demote: true }),
  // English title page, left-to-right, as the closing page of the volume.
  new D.Paragraph({ children: [new D.PageBreak()] }),
  blank(600),
  centered("University of Isfahan", 30, true, 140, true),
  centered("Faculty of Engineering", 26, true, 100, true),
  centered("Department of Biomedical Engineering", 24, true, 900, true),
  centered("B.Sc. Project", 32, true, 700, true),
  centered("Design and Implementation of an Intelligent Automated", 30, true, 60, true),
  centered("Antibiogram Analysis System Based on Image Processing", 30, true, 60, true),
  centered("and Machine Learning According to EUCAST Standards", 30, true, 900, true),
  centered("Supervisor:", 28, true, 90, true),
  centered("Dr. Mohammad Reza Yazdchi", 28, true, 600, true),
  centered("By:", 28, true, 90, true),
  centered("Mahyar Haddadha", 28, true, 900, true),
  centered("September 2026", 26, true, 160, true),
];
const body = ONLY ? pieces.slice(0, ONLY) : pieces;

// ---- References, rendered as an IEEE numbered list ----
function refsSection() {
  const md = read("references.md");
  const rows = [...md.matchAll(/^\|\s*\[(\d+)\]\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$/gm)];
  const lines = [
    "ارجاع‌دهی در این نوشتار بر پایه‌ی سبک IEEE و به ترتیب اولین ظهور در متن انجام شده است. اطلاعات کتاب‌شناختی منابع علامت‌خورده، بر پایه‌ی مقالات بازیابی‌شده از پایگاه PubMed راستی‌آزمایی شده و شناسه‌ی PMID و DOI هر مورد ثبت شده است تا هر ارجاع مستقلاً قابل بررسی باشد.",
    "",
  ];
  rows.forEach((r) => {
    const doi = r[4].match(/\((https?:\/\/doi\.org\/[^)]+)\)/);
    lines.push(`[${r[1]}] ${r[2].replace(/\*/g, "")}${doi ? "  " + doi[1] : ""}`);
  });
  return lines.join("\n");
}

const doc = new D.Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: SZ.body }, paragraph: { spacing: { line: 300 } } },
    },
  },
  sections: [{
    properties: { page: PAGE, bidi: true },
    footers: {
      default: new D.Footer({
        children: [new D.Paragraph({
          alignment: D.AlignmentType.CENTER,
          bidirectional: true,
          children: [new D.TextRun({
            font: FONT, size: SZ.tbl, rightToLeft: true,
            children: [D.PageNumber.CURRENT],
          })],
        })],
      }),
    },
    children: body,
  }],
});

// docx-js does not emit the section-level RTL flag, which the university
// template carries, so it is inserted into sectPr after packing.
function addSectionRtl(buf) {
  const AdmZip = require("adm-zip");
  const zip = new AdmZip(buf);
  const entry = "word/document.xml";
  let xml = zip.readAsText(entry);
  // CT_SectPr is a fixed sequence: w:bidi belongs after w:cols and
  // immediately before w:docGrid, not at the head of the element.
  if (!/<w:bidi\/>\s*<w:docGrid/.test(xml)) {
    xml = xml
      .replace(/<w:sectPr\b([^>]*)><w:bidi\/>/g, "<w:sectPr$1>")
      .replace(/(<w:sectPr\b[^>]*>(?:(?!<\/w:sectPr>)[\s\S])*?)(<w:docGrid)/g, "$1<w:bidi/>$2");
    zip.updateFile(entry, Buffer.from(xml, "utf8"));
  }
  return zip.toBuffer();
}

D.Packer.toBuffer(doc).then((buf) => {
  const out = path.join(THESIS, ONLY ? `test_${ONLY}.docx` : "Antibiogram_Thesis_Haddadha.docx");
  const final = addSectionRtl(buf);
  fs.writeFileSync(out, final);
  console.log("wrote", out, "blocks:", body.length, (final.length / 1024).toFixed(0) + " KB");
});
