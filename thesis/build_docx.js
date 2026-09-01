#!/usr/bin/env node
/**
 * Build the final Persian thesis .docx from the markdown chapters.
 *
 * Layout follows thesis_template.docx: A4, RTL, IRNazanin, 1985/1701 twip
 * margins, chapter-scoped figure and table numbering, IEEE references.
 *
 * Footnote markers and reference numbers are rendered with Persian digits;
 * the footnote text and the reference entries themselves stay in English.
 *
 *   node thesis/build_docx.js
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");

const THESIS = __dirname;
const REPO = path.dirname(THESIS);
const FIGS = path.join(THESIS, "figures");
const GALLERY = path.join(FIGS, "pipeline", "gt_06");

// ─── Template constants ────────────────────────────────────────────────
const FONT = "IRNazanin";
const FONT_EN = "Times New Roman";
const CONTENT_W = 11906 - 1985 * 2; // usable width in DXA

const SZ = { body: 26, h1: 30, h2: 28, h3: 26, cap: 24, tbl: 22, small: 20, fn: 20 };

const PAGE = {
  size: { width: 11906, height: 16838 },
  margin: { top: 1701, right: 1985, bottom: 1701, left: 1985 },
};

const toFa = (n) => String(n).replace(/\d/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]);

// ─── Footnote state ────────────────────────────────────────────────────
// Markdown carries per-file keys ([^1], [^2] ...); the document needs one
// continuous sequence, so keys are remapped as each file is converted.
const footnotes = {};           // docx id -> { children: [Paragraph] }
let footnoteSeq = 0;
let footnoteDefs = {};          // per-file key -> English text
let footnoteIds = {};           // per-file key -> docx id

function footnoteId(key) {
  if (footnoteIds[key]) return footnoteIds[key];
  const text = footnoteDefs[key];
  if (!text) return null;
  const id = ++footnoteSeq;
  footnoteIds[key] = id;
  footnotes[id] = {
    children: [new D.Paragraph({
      alignment: D.AlignmentType.LEFT,
      bidirectional: false,
      spacing: { after: 0, line: 240 },
      children: [
        new D.TextRun({ text: "  ", font: FONT_EN, size: SZ.fn }),
        new D.TextRun({ text, font: FONT_EN, size: SZ.fn, rightToLeft: false }),
      ],
    })],
  };
  return id;
}

// ─── Inline formatting ─────────────────────────────────────────────────
// Splits on **bold**, *italic*, `code`, $math$ and [^n] footnote markers.
function runs(text, opt = {}) {
  const size = opt.size || SZ.body;
  const base = { font: FONT, size, rightToLeft: true, ...(opt.run || {}) };
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\$[^$]+\$|\[\^[^\]]+\])/g;
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
    if (tok.startsWith("[^")) {
      const id = footnoteId(tok.slice(2, -1));
      if (id) out.push(new D.FootnoteReferenceRun(id));
    } else if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
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
// Explicit, because the reading order of the chapter is not the execution
// order of the notebook cells the gallery images come from.
const GALLERY_FOR = {
  1: "01_input_image", 2: "02_dish_detection", 3: "03_dish_mask",
  4: "04_tophat_a", 5: "05_tophat_b", 6: "06_threshold", 7: "07_closing",
  8: "08_opening", 9: "09_distance_transform", 10: "14_watershed_markers",
  11: "10_halo_gradient", 12: "11_disk_edges", 13: "12_hough_candidates",
  14: "13_blob_watershed", 15: "15_disks_final", 16: "16_agar_canvas",
  17: "21_halo_base", 18: "22_halo_growth", 19: "23_halo_angular_fix",
  20: "17_branch_otsu", 21: "18_branch_watershed", 22: "19_branch_statistical",
  23: "20_branch_growth_model", 24: "24_halo_fusion", 25: "25_bubbles",
  26: "26_eucast", 27: "27_final_report",
};

const FIGURES = { "۳-۱": path.join(FIGS, "fig_3_1_architecture.png") };
for (const [n, file] of Object.entries(GALLERY_FOR)) {
  FIGURES[`۴-${toFa(n)}`] = path.join(GALLERY, `${file}.png`);
}
[["۵-۱", "fig_5_1_bland_altman.png"],
 ["۵-۲", "fig_5_2_system_vs_expert.png"],
 ["۵-۳", "fig_5_3_branch_tradeoff.png"],
 ["۵-۴", "fig_5_4_error_distribution.png"],
 ["۵-۵", "fig_5_5_error_vs_size.png"],
 ["۵-۶", "fig_5_6_false_positives.png"],
 ["۵-۷", "fig_5_7_per_image.png"],
 ["۵-۸", "fig_5_8_best_case.png"],
 ["۵-۹", "fig_5_9_worst_case.png"],
 ["۵-۱۰", "fig_5_10_clinical_target.png"],
 ["۶-۱", "fig_6_1_separability.png"],
 ["۶-۲", "fig_6_2_filter_study.png"],
 ["۶-۳", "fig_6_3_sigmoid_projection.png"],
].forEach(([k, f]) => { FIGURES[k] = path.join(FIGS, f); });

// ─── Markdown → docx blocks ────────────────────────────────────────────
function convert(md, opts = {}) {
  const out = [];

  // Footnote definitions are collected first, then dropped from the body.
  footnoteDefs = {};
  footnoteIds = {};
  const lines = [];
  md.split("\n").forEach((l) => {
    const m = l.match(/^\[\^([^\]]+)\]:\s*(.*)$/);
    if (m) footnoteDefs[m[1]] = m[2].trim();
    else lines.push(l);
  });

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (!t) { i++; continue; }
    if (/^---+$/.test(t)) { i++; continue; }              // horizontal rule
    if (/^<\/?div/.test(t)) { i++; continue; }            // html wrappers

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

    // Display maths, written as a $$…$$ line of its own
    if (/^\$\$.*\$\$$/.test(t)) {
      out.push(new D.Paragraph({
        children: [new D.TextRun({
          text: t.slice(2, -2).trim(), font: FONT_EN, size: SZ.body,
          italics: true, rightToLeft: false,
        })],
        alignment: D.AlignmentType.CENTER,
        bidirectional: false,
        spacing: { before: 140, after: 160 },
      }));
      i++; continue;
    }

    // Figure caption lines → image + caption
    if ((m = t.match(/^\*{1,2}(?:شکلِ?|شکل)\s*([۰-۹]+-[۰-۹]+)\s*[:：](.*)$/))) {
      const num = m[1];
      const capText = ("شکل " + num + ": " + m[2]).replace(/\*+$/, "").trim();
      const f = FIGURES[num];
      if (f && fs.existsSync(f)) out.push(image(f, num.startsWith("۴") ? 320 : 420));
      else console.warn("  missing figure", num, f);
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
      if (!n || /^[#>|`-]/.test(n) || /^\$\$/.test(n) || /^\d+[.)]\s/.test(n)
          || /^[۰-۹]+[.)]\s/.test(n) || /^\*{1,2}شکل/.test(n) || /^<\/?div/.test(n)) break;
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

// ---- References, rendered as an IEEE numbered list ----
// The bracket number is Persian, to match the in-text citations; the entry
// itself stays English and left to right.
function referenceParagraphs() {
  const md = read("references.md");
  const rows = [...md.matchAll(/^\|\s*\[(\d+)\]\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$/gm)];
  const out = [para(
    "ارجاع‌دهی در این نوشتار بر پایه سبک IEEE و به ترتیب نخستین ظهور در متن انجام شده است. " +
    "اطلاعات کتاب‌شناختی منابع نمایه‌شده در PubMed با جستجوی مستقیم عنوان در آن پایگاه راستی‌آزمایی " +
    "شده و شناسه PMID و DOI هر مورد ثبت گردیده است، تا هر ارجاع مستقلاً قابل بررسی بماند."
  ), blank(200)];

  rows.forEach((r) => {
    // The DOI is taken from the link *text*, because DOI strings may
    // themselves contain brackets that truncate a URL-side match.
    const doi = r[4].match(/^\[(10\.[^\]]+)\]/);
    const body = r[2].replace(/\*/g, "") + (doi ? "  https://doi.org/" + doi[1] : "");
    out.push(new D.Paragraph({
      alignment: D.AlignmentType.LEFT,
      bidirectional: false,
      spacing: { after: 130, line: 280 },
      indent: { left: 480, hanging: 480 },
      children: [
        new D.TextRun({ text: `[${toFa(r[1])}]`, font: FONT, size: SZ.body }),
        new D.TextRun({ text: "  " + body, font: FONT_EN, size: SZ.tbl, rightToLeft: false }),
      ],
    }));
  });
  return out;
}

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

const body = [
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
  ...referenceParagraphs(),
  ...convert(read("appendix_a_evaluation_data.md")),
  // English title page, left to right, as the closing page of the volume.
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

const doc = new D.Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: SZ.body }, paragraph: { spacing: { line: 300 } } },
    },
  },
  footnotes,
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

// docx-js does not emit the section-level RTL flag that the university
// template carries, and Word has no footnote numbering format that yields
// Persian digits, so both are patched into the package after packing.
function postProcess(buf) {
  const AdmZip = require("adm-zip");
  const zip = new AdmZip(buf);

  let xml = zip.readAsText("word/document.xml");
  // CT_SectPr is a fixed sequence: w:bidi belongs after w:cols and
  // immediately before w:docGrid, not at the head of the element.
  if (!/<w:bidi\/>\s*<w:docGrid/.test(xml)) {
    xml = xml
      .replace(/<w:sectPr\b([^>]*)><w:bidi\/>/g, "<w:sectPr$1>")
      .replace(/(<w:sectPr\b[^>]*>(?:(?!<\/w:sectPr>)[\s\S])*?)(<w:docGrid)/g, "$1<w:bidi/>$2");
  }
  // Replace each automatic footnote mark with an explicit Persian numeral.
  xml = xml.replace(
    /<w:r><w:rPr><w:rStyle w:val="FootnoteReference"\/><\/w:rPr><w:footnoteReference w:id="(\d+)"\/><\/w:r>/g,
    (_, id) =>
      '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/>' +
      `<w:rFonts w:ascii="${FONT}" w:cs="${FONT}" w:hAnsi="${FONT}"/>` +
      '<w:vertAlign w:val="superscript"/></w:rPr>' +
      `<w:footnoteReference w:customMarkFollows="1" w:id="${id}"/>` +
      `<w:t>${toFa(id)}</w:t></w:r>`);
  zip.updateFile("word/document.xml", Buffer.from(xml, "utf8"));

  let fx = zip.readAsText("word/footnotes.xml");
  fx = fx.replace(
    /(<w:footnote w:id="(\d+)">(?:(?!<\/w:footnote>)[\s\S])*?)<w:footnoteRef\/>/g,
    (whole, head, id) => Number(id) > 0
      ? head + `<w:t>${toFa(id)}</w:t>`
      : whole);
  zip.updateFile("word/footnotes.xml", Buffer.from(fx, "utf8"));

  return zip.toBuffer();
}

D.Packer.toBuffer(doc).then((buf) => {
  const out = path.join(THESIS, "Antibiogram_Thesis_Haddadha.docx");
  const final = postProcess(buf);
  fs.writeFileSync(out, final);
  console.log("wrote", out,
              "| blocks:", body.length,
              "| footnotes:", footnoteSeq,
              "|", (final.length / 1024 / 1024).toFixed(2) + " MB");
});
