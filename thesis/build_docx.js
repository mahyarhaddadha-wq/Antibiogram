#!/usr/bin/env node
/**
 * Build the final Persian thesis .docx inside the container the faculty
 * supplies as thesis_template.docx (the file the author calls «قالب»).
 *
 * Everything below is read out of that template rather than invented:
 *
 *   page      11906 x 16838, margins 1701/1985/1701/1985, header and
 *             footer 720, no gutter, bidi with an rtlGutter
 *   body      IRNazanin 13 (Latin runs Times New Roman 12)
 *   headings  IRNazanin bold, 15 / 14 / 13 for levels one to three
 *   captions  the template's own «pic» and «table» paragraph styles,
 *             IRNazanin 12 bold, centred, and with no colon after the
 *             number, exactly as «شکل 1-1 آرم دانشگاه اصفهان»
 *   cover     IRNazanin 13/12/11 for the university, faculty and group,
 *             18 for the degree line and IRTitr 13 for the title
 *   abstract  Times New Roman 12, its heading bold
 *   numbering front matter unnumbered, the three lists in Arabic letters,
 *             the body in digits from one
 *   lists     فهرست مطالب is a TOC field over heading levels one to
 *             three; فهرست شکل‌ها and فهرست جدول‌ها are built from the
 *             captions themselves, each entry a hyperlink to its caption
 *             with a dot leader and a PAGEREF field for the page number
 *   sources   IEEE, which is what the template prescribes for engineering
 *
 *   node thesis/build_docx.js
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");

const THESIS = __dirname;
const FIGS = path.join(THESIS, "figures");
const GALLERY = path.join(FIGS, "pipeline", "gt_06");

// ─── Template constants ────────────────────────────────────────────────
const FA = "IRNazanin";        // Persian body and heading face
const FA_TITLE = "IRTitr";     // Persian display face, cover title only
const EN = "Times New Roman";  // Latin face

const FACE = { ascii: EN, hAnsi: EN, eastAsia: EN, cs: FA };
const FACE_TITLE = { ascii: EN, hAnsi: EN, eastAsia: EN, cs: FA_TITLE };

// [Latin size, complex-script size] in half points. The complex-script
// value is the size annotated in the template; the Latin value is one
// point smaller, because Times New Roman sets larger than IRNazanin.
const SZ = {
  body: [24, 26], h1: [28, 30], h2: [26, 28], h3: [24, 26],
  cap: [22, 24], tbl: [22, 24], note: [20, 20], code: [18, 18],
  en: [24, 24],
};

const PAGE = {
  size: { width: 11906, height: 16838 },
  margin: { top: 1701, right: 1985, bottom: 1701, left: 1985,
            header: 720, footer: 720, gutter: 0 },
};

const CONTENT_W = 11906 - 1985 * 2;  // 7936 dxa of usable width
const TOC_TAB = 7926;                // the template's dot-leader stop
const LINE = 259;                    // the template's default line spacing
const FIRST_LINE = 284;              // 0.5 cm first-line indent

const FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
const toFa = (n) => String(n).replace(/\d/g, (d) => FA_DIGITS[d]);
const isLatin = (t) => !/[\u0600-\u06FF]/.test(t);

// ─── Footnote state ────────────────────────────────────────────────────
// Markdown carries per-file keys ([^1], [^2] ...); the document needs one
// continuous sequence, so keys are remapped as each file is converted.
// Word restarts the printed marks on every page by itself.
const footnotes = {};
let footnoteSeq = 0;
let footnoteDefs = {};
let footnoteIds = {};

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
      spacing: { after: 0, line: 240, lineRule: D.LineRuleType.AUTO },
      indent: { left: 170, hanging: 170 },
      children: [new D.TextRun({
        text: " " + text, font: FACE,
        size: SZ.note[0], sizeComplexScript: SZ.note[1], rightToLeft: false,
      })],
    })],
  };
  return id;
}

// ─── Inline formatting ─────────────────────────────────────────────────
// Splits on **bold**, *italic*, `code`, $math$ and [^n] footnote markers.
// Bold spans are flattened to plain text in running prose, because the
// author asked for as few bold sentences as possible; bold survives only
// where the template itself demands it, in captions and table headers.
function runs(text, opt = {}) {
  const [sz, szCs] = opt.size || SZ.body;
  const face = opt.face || FACE;
  const rtl = opt.rtl !== false;
  const keepBold = !!opt.keepBold;
  const base = { font: face, size: sz, sizeComplexScript: szCs,
                 rightToLeft: rtl, ...(opt.run || {}) };
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\$[^$]+\$|\[\^[^\]]+\])/g;
  let last = 0, m;
  const push = (t, extra) => {
    if (!t) return;
    out.push(new D.TextRun({
      ...base, text: t, ...extra,
      ...(rtl && isLatin(t) ? { rightToLeft: false } : {}),
    }));
  };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("[^")) {
      const id = footnoteId(tok.slice(2, -1));
      if (id) out.push(new D.FootnoteReferenceRun(id));
    } else if (tok.startsWith("**")) {
      out.push(...runs(tok.slice(2, -2),
        { ...opt, run: { ...(opt.run || {}), ...(keepBold ? { bold: true } : {}) } }));
    } else if (tok.startsWith("`")) {
      push(tok.slice(1, -1), {
        font: { ascii: "Consolas", hAnsi: "Consolas", cs: "Consolas" },
        size: SZ.code[0], sizeComplexScript: SZ.code[1], rightToLeft: false });
    } else if (tok.startsWith("$")) {
      push(tok.slice(1, -1), { italics: true, rightToLeft: false });
    } else {
      out.push(...runs(tok.slice(1, -1),
        { ...opt, run: { ...(opt.run || {}), italics: true } }));
    }
    last = re.lastIndex;
  }
  push(text.slice(last));
  return out.length ? out : [new D.TextRun({ ...base, text: "" })];
}

// A body paragraph: right to left, justified, first line indented.
function para(text, o = {}) {
  const latin = isLatin(text.replace(/[*`$\[\]^]/g, "").trim());
  return new D.Paragraph({
    children: runs(text, { ...o, rtl: !latin }),
    bidirectional: !latin,
    alignment: o.alignment || (latin ? D.AlignmentType.LEFT : D.AlignmentType.BOTH),
    spacing: { before: o.before ?? 0, after: o.after ?? 120,
               line: LINE, lineRule: D.LineRuleType.AUTO },
    indent: o.indent !== undefined ? o.indent : { firstLine: FIRST_LINE },
    ...(o.border ? { border: o.border } : {}),
  });
}

const blank = (after = 0) => new D.Paragraph({
  text: "", spacing: { after, line: LINE, lineRule: D.LineRuleType.AUTO },
});

// ─── Headings ──────────────────────────────────────────────────────────
// In a right-to-left paragraph w:jc="left" is the leading edge, so these
// sit against the right margin, which is where the template has them.
function heading(text, level, o = {}) {
  const cfg = {
    1: { size: SZ.h1, heading: D.HeadingLevel.HEADING_1, before: 0, after: 240 },
    2: { size: SZ.h2, heading: D.HeadingLevel.HEADING_2, before: 280, after: 80 },
    3: { size: SZ.h3, heading: D.HeadingLevel.HEADING_3, before: 200, after: 60 },
  }[level];
  return new D.Paragraph({
    children: [new D.TextRun({
      font: FACE, size: cfg.size[0], sizeComplexScript: cfg.size[1],
      bold: true, rightToLeft: true, text,
    })],
    heading: cfg.heading,
    bidirectional: true,
    alignment: D.AlignmentType.LEFT,
    keepNext: true,
    spacing: { before: cfg.before, after: cfg.after, line: LINE, lineRule: D.LineRuleType.AUTO },
    ...(level === 1 && !o.noBreak ? { pageBreakBefore: true } : {}),
  });
}

// Front-matter and list titles. They are deliberately not headings, so
// that the table of contents lists only the chapters, the reference list,
// the glossary and the appendix, exactly as the template's own does.
const plainTitle = (text, o = {}) => {
  const latin = isLatin(text);
  return new D.Paragraph({
    children: [new D.TextRun({
      font: FACE, size: latin ? SZ.en[0] : SZ.h3[0],
      sizeComplexScript: latin ? SZ.en[1] : SZ.h3[1],
      bold: true, rightToLeft: !latin, text,
    })],
    bidirectional: !latin,
    alignment: D.AlignmentType.CENTER,
    pageBreakBefore: o.pageBreakBefore !== false,
    spacing: { before: 0, after: 280, line: LINE, lineRule: D.LineRuleType.AUTO },
  });
};

// ─── Captions ──────────────────────────────────────────────────────────
// The template numbers captions without a colon and carries two paragraph
// styles for them, «pic» below a figure and «table» above a table. Each
// caption is bookmarked so the two lists can point a PAGEREF at it.
const CAPTIONS = [];   // {kind, num, text, bm} in reading order

let captionSeq = 0;

function caption(kind, num, text) {
  const bm = (kind === "fig" ? "fig_" : "tab_") + ++captionSeq;
  const label = (kind === "fig" ? "شکل " : "جدول ") + num + " " + text;
  CAPTIONS.push({ kind, num, text, bm, label });
  return new D.Paragraph({
    style: kind === "fig" ? "pic" : "table",
    children: [new D.Bookmark({
      id: bm,
      children: runs(label, { size: SZ.cap, run: { bold: true } }),
    })],
    keepNext: kind === "tab",
  });
}

// One entry of فهرست شکل‌ها or فهرست جدول‌ها: the caption text, a dot
// leader, and the page the caption actually lands on.
const listEntry = (c) => new D.Paragraph({
  bidirectional: true,
  alignment: D.AlignmentType.LEFT,
  spacing: { after: 100, line: LINE, lineRule: D.LineRuleType.AUTO },
  tabStops: [{ type: D.TabStopType.RIGHT, position: TOC_TAB, leader: D.LeaderType.DOT }],
  children: [new D.InternalHyperlink({
    anchor: c.bm,
    children: [
      ...runs(c.label, { size: SZ.body }),
      new D.TextRun({ children: ["\t"], font: FACE,
                      size: SZ.body[0], sizeComplexScript: SZ.body[1] }),
      new D.PageReference(c.bm, { hyperlink: false }),
    ],
  })],
});

// ─── Images ────────────────────────────────────────────────────────────
function image(file, maxWidthPt = 400) {
  const buf = fs.readFileSync(file);
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
    keepNext: true,
    spacing: { before: 240, after: 60, line: LINE, lineRule: D.LineRuleType.AUTO },
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
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      verticalAlign: D.VerticalAlign.CENTER,
      children: [new D.Paragraph({
        children: runs(cells[i] ?? "", { size: SZ.tbl, run: { bold: !!isHead } }),
        bidirectional: true,
        alignment: D.AlignmentType.CENTER,
        spacing: { before: 0, after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
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

// ─── Numbered formulas ─────────────────────────────────────────────────
// A numbered formula is a one-row table: the expression in the wide cell
// and its number in a narrow cell at the right, which is the leading edge
// of a right-to-left row. The markdown keeps the LaTeX so the source
// stays readable; what Word receives is the plain-Unicode transcription
// below, which the author can replace with a real Word equation.
const FORMULA = {
  "۲-۱": ["σ²b(t) = ω₀(t) · ω₁(t) · [ μ₀(t) − μ₁(t) ]²"],
  "۲-۲": ["I_corr = I ⁄ I_illum", "I_corr = I − I_illum + Ī_illum"],
  "۲-۳": ["MAD(X) = median( | xᵢ − median(X) | )"],
  "۲-۴": ["d = ( μ_region − μ_background ) ⁄ σ_background"],
  "۲-۵": ["dᵢ = yᵢ − xᵢ", "Bias = d̄", "LoA = d̄ ± 1.96 · s_d"],
  "۲-۶": ["MAE = (1 ⁄ n) · Σᵢ₌₁ⁿ | yᵢ − xᵢ |"],
  "۶-۱": ["y(r) = A + ( B − A ) ⁄ ( 1 + exp( −( r − r₀ ) ⁄ w ) )"],
  "۶-۲": ["δ = 2 · k · w ⁄ p", "k = ln( 0.95 ⁄ 0.05 ) ≈ 2.944"],
};

function formula(num, latex) {
  const lines = FORMULA[num] || [latex];
  const numW = 900, bodyW = CONTENT_W - numW;
  const cell = (children, w) => new D.TableCell({
    width: { size: w, type: D.WidthType.DXA },
    borders: {
      top: { style: D.BorderStyle.NONE }, bottom: { style: D.BorderStyle.NONE },
      left: { style: D.BorderStyle.NONE }, right: { style: D.BorderStyle.NONE },
    },
    margins: { top: 60, bottom: 60, left: 70, right: 70 },
    verticalAlign: D.VerticalAlign.CENTER,
    children,
  });
  return new D.Table({
    columnWidths: [numW, bodyW],
    width: { size: CONTENT_W, type: D.WidthType.DXA },
    visuallyRightToLeft: true,
    borders: {
      top: { style: D.BorderStyle.NONE }, bottom: { style: D.BorderStyle.NONE },
      left: { style: D.BorderStyle.NONE }, right: { style: D.BorderStyle.NONE },
      insideHorizontal: { style: D.BorderStyle.NONE },
      insideVertical: { style: D.BorderStyle.NONE },
    },
    rows: [new D.TableRow({
      cantSplit: true,
      children: [
        cell([new D.Paragraph({
          bidirectional: true,
          alignment: D.AlignmentType.CENTER,
          spacing: { before: 0, after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
          children: [new D.TextRun({
            text: `(${num})`, font: FACE,
            size: SZ.body[0], sizeComplexScript: SZ.body[1], rightToLeft: true,
          })],
        })], numW),
        cell(lines.map((l) => new D.Paragraph({
          bidirectional: false,
          alignment: D.AlignmentType.CENTER,
          spacing: { before: 0, after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
          children: [new D.TextRun({
            text: l, font: FACE, italics: true,
            size: SZ.body[0], sizeComplexScript: SZ.body[1], rightToLeft: false,
          })],
        })), bodyW),
      ],
    })],
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

  footnoteDefs = {};
  footnoteIds = {};
  const lines = [];
  md.split("\n").forEach((l) => {
    const m = l.match(/^\[\^([^\]]+)\]:\s*(.*)$/);
    if (m) footnoteDefs[m[1]] = m[2].trim();
    else lines.push(l);
  });

  let i = 0;
  let pendingCaption = null;   // a table caption waits for its table
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (!t) { i++; continue; }
    if (/^---+$/.test(t)) { i++; continue; }
    if (/^<\/?div/.test(t)) { i++; continue; }

    let m;
    if ((m = t.match(/^(#{1,4})\s+(.*)$/))) {
      const depth = m[1].length;
      const text = m[2].replace(/\*\*/g, "").trim();
      if ((opts.skip || []).some((k) => text.includes(k))) {
        i++;
        while (i < lines.length && !new RegExp(`^#{1,${depth}} `).test(lines[i])) i++;
        continue;
      }
      if (depth === 1) {
        const noBreak = !!opts.noFirstBreak && !out.length;
        out.push(opts.plainH1 ? plainTitle(text, { pageBreakBefore: !noBreak })
                              : heading(text, 1, { noBreak }));
      }
      else if (depth === 2) out.push(heading(text, opts.demote ? 3 : 2));
      else out.push(heading(text, 3));
      i++; continue;
    }

    if (t.startsWith("```")) {
      const body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) body.push(lines[i]), i++;
      i++;
      body.forEach((b) => out.push(new D.Paragraph({
        children: [new D.TextRun({
          font: { ascii: "Consolas", hAnsi: "Consolas", cs: "Consolas" },
          size: SZ.code[0], sizeComplexScript: SZ.code[1], text: b || " ", rightToLeft: false,
        })],
        alignment: D.AlignmentType.LEFT,
        spacing: { after: 0, line: 240, lineRule: D.LineRuleType.AUTO },
        bidirectional: false,
      })));
      out.push(blank(120));
      continue;
    }

    // A table caption sits above its table, as the template shows.
    if ((m = t.match(/^\*\*جدول\s*((?:[۰-۹]+|الف|ب|پ|ت|ث)-[۰-۹]+)\s*[:：]?\s*(.*?)\*\*$/))) {
      pendingCaption = caption("tab", m[1], m[2].trim());
      i++; continue;
    }

    if (t.startsWith("|") && (lines[i + 1] || "").trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        if (!/^[\s:\-|]+$/.test(cells.join(""))) rows.push(cells);
        i++;
      }
      if (rows.length) {
        if (pendingCaption) { out.push(pendingCaption); pendingCaption = null; }
        out.push(table(rows));
        out.push(blank(200));
      }
      continue;
    }
    if (pendingCaption) { out.push(pendingCaption); pendingCaption = null; }

    // A display formula, optionally followed by its number.
    if ((m = t.match(/^\$\$([\s\S]*?)\$\$\s*(?:\(رابطه\s*([۰-۹]+-[۰-۹]+)\))?$/))) {
      if (m[2]) { out.push(formula(m[2], m[1].trim())); out.push(blank(120)); }
      else out.push(new D.Paragraph({
        children: [new D.TextRun({
          text: m[1].trim(), font: FACE, italics: true,
          size: SZ.body[0], sizeComplexScript: SZ.body[1], rightToLeft: false,
        })],
        alignment: D.AlignmentType.CENTER,
        bidirectional: false,
        spacing: { before: 140, after: 160, line: LINE, lineRule: D.LineRuleType.AUTO },
      }));
      i++; continue;
    }

    // A figure caption sits below its figure, as the template shows.
    if ((m = t.match(/^\*{1,2}(?:شکلِ?|شکل)\s*((?:[۰-۹]+|الف|ب|پ|ت|ث)-[۰-۹]+)\s*[:：]?\s*(.*)$/))) {
      const num = m[1];
      const f = FIGURES[num];
      if (f && fs.existsSync(f)) out.push(image(f, num.startsWith("۴") ? 320 : 420));
      else console.warn("  missing figure", num, f);
      out.push(caption("fig", num, m[2].replace(/\*+$/, "").trim()));
      i++; continue;
    }

    if (t.startsWith(">")) {
      const body = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        body.push(lines[i].trim().replace(/^>\s?/, "")); i++;
      }
      body.filter((b) => b.trim()).forEach((b) => out.push(para(b, {
        indent: { left: 340, firstLine: 0 },
        border: { right: { style: D.BorderStyle.SINGLE, size: 12, color: "808080", space: 10 } },
        after: 60,
      })));
      out.push(blank(120));
      continue;
    }

    if (/^[-*]\s+/.test(t) || /^\d+[.)]\s+/.test(t) || /^[۰-۹]+[.)]\s+/.test(t)) {
      out.push(para(t.replace(/^[-*]\s+/, "• ").replace(/^([۰-۹\d]+[.)])\s+/, "$1 "), {
        indent: { left: 340, hanging: 180 },
        after: 60, keepBold: opts.keepBold,
      }));
      i++; continue;
    }

    const buf = [t];
    i++;
    while (i < lines.length) {
      const n = lines[i].trim();
      if (!n || /^[#>|`-]/.test(n) || /^\$\$/.test(n) || /^\d+[.)]\s/.test(n)
          || /^[۰-۹]+[.)]\s/.test(n) || /^\*{1,2}شکل/.test(n) || /^\*\*جدول/.test(n)
          || /^<\/?div/.test(n)) break;
      buf.push(n); i++;
    }
    out.push(para(buf.join(" "), { keepBold: opts.keepBold }));
  }
  if (pendingCaption) out.push(pendingCaption);
  return out;
}

// ─── Document assembly ─────────────────────────────────────────────────
const read = (f) => fs.readFileSync(path.join(THESIS, f), "utf8");

// A centred cover line. The sizes are the ones written into the template.
const cover = (text, pt, o = {}) => new D.Paragraph({
  children: [new D.TextRun({
    font: o.titr ? FACE_TITLE : (o.en ? FACE : FACE),
    size: o.en ? pt * 2 : pt * 2 - 2,
    sizeComplexScript: pt * 2,
    bold: o.bold !== false,
    rightToLeft: !o.en,
    text,
  })],
  alignment: D.AlignmentType.CENTER,
  bidirectional: !o.en,
  spacing: { after: o.after ?? 160, line: LINE, lineRule: D.LineRuleType.AUTO },
});

// ---- References: left to right, English, hanging indent, Latin numbers
function referenceParagraphs() {
  const md = read("references.md");
  const rows = [...md.matchAll(/^\|\s*\[(\d+)\]\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$/gm)];
  const out = [para(
    "ارجاع‌دهی این نوشتار بر پایه سبک IEEE و به ترتیب نخستین ظهور در متن انجام شده است، همان " +
    "سبکی که قالب دانشکده برای رشته‌های مهندسی تعیین کرده است. شماره ارجاع در متن فارسی و درون " +
    "کروشه فارسی آمده است و مشخصات هر منبع در همین فهرست به انگلیسی و درون کروشه انگلیسی نوشته " +
    "شده است. اطلاعات کتاب‌شناختی منابع نمایه‌شده در PubMed با جستجوی مستقیم عنوان در آن پایگاه " +
    "راستی‌آزمایی شده و شناسه PMID و DOI هر مورد ثبت گردیده است، تا هر ارجاع مستقلاً قابل بررسی بماند."
  ), blank(200)];

  rows.forEach((r) => {
    const doi = r[4].match(/^\[(10\.[^\]]+)\]/);
    const body = `[${r[1]}]\t` + r[2].replace(/\*/g, "") +
                 (doi ? "  https://doi.org/" + doi[1] : "");
    out.push(new D.Paragraph({
      alignment: D.AlignmentType.LEFT,
      bidirectional: false,
      spacing: { after: 120, line: LINE, lineRule: D.LineRuleType.AUTO },
      indent: { left: 425, hanging: 425 },
      children: [new D.TextRun({
        text: body, font: FACE,
        size: SZ.en[0], sizeComplexScript: SZ.en[1], rightToLeft: false,
      })],
    }));
  });
  return out;
}

// ---- Cover pages ----
const titlePage = [
  blank(500),
  cover("دانشگاه اصفهان", 13),
  cover("دانشکده فنی و مهندسی", 12),
  cover("گروه مهندسی پزشکی", 11, { after: 900 }),
  cover("پروژه‌ی کارشناسی رشته‌ی مهندسی پزشکی", 18, { after: 800 }),
  cover("طراحی و پیاده‌سازی سامانه هوشمند تحلیل خودکار آزمون آنتی‌بیوگرام", 13, { titr: true, after: 80 }),
  cover("مبتنی بر پردازش تصویر و یادگیری ماشین مطابق استاندارد EUCAST", 13, { titr: true, after: 900 }),
  cover("استاد راهنما:", 15, { after: 80 }),
  cover("دکتر محمدرضا یزدچی", 13, { after: 600 }),
  cover("دانشجو:", 15, { after: 80 }),
  cover("مهیار حدادها", 13, { after: 60 }),
  cover("۴۰۱۲۰۱۳۰۵۳", 13, { after: 900 }),
  cover("شهریور ۱۴۰۵", 13),
];

const englishTitlePage = [
  blank(500),
  cover("University of Isfahan", 12, { en: true, after: 120 }),
  cover("Faculty of Engineering", 11, { en: true, after: 100 }),
  cover("Department of Biomedical Engineering", 10, { en: true, after: 900 }),
  cover("B.Sc. Project", 16, { en: true, after: 800 }),
  cover("Design and Implementation of an Intelligent Automated", 15, { en: true, after: 60 }),
  cover("Antibiogram Analysis System Based on Image Processing", 15, { en: true, after: 60 }),
  cover("and Machine Learning According to EUCAST Standards", 15, { en: true, after: 900 }),
  cover("Supervisor:", 14, { en: true, after: 80 }),
  cover("Dr. Mohammad Reza Yazdchi", 13, { en: true, after: 600 }),
  cover("By:", 14, { en: true, after: 80 }),
  cover("Mahyar Haddadha", 13, { en: true, after: 900 }),
  cover("September 2026", 13, { en: true }),
];

// ---- Front matter, split so the page numbering can follow the template
const front = read("front_matter.md");
const cutLists = front.indexOf("# فهرست مطالب");
const cutGlossary = front.indexOf("# واژه‌نامه");
const cutEnglish = front.indexOf("# Abstract");
const frontA = front.slice(0, cutLists);            // pledge, thanks, abstract
const glossary = front.slice(cutGlossary, cutEnglish);
const frontC = front.slice(cutEnglish);             // English abstract

// ---- The body is converted first, so the two lists know their captions
const bodyBlocks = [
  ...convert(read("chapter_01.md"), { noFirstBreak: true }),
  ...convert(read("chapter_02.md")),
  ...convert(read("chapter_03.md")),
  ...convert(read("chapter_04.md")),
  ...convert(read("chapter_05.md")),
  ...convert(read("chapter_06.md")),
];
const backBlocks = [
  heading("منابع و مآخذ", 1),
  ...referenceParagraphs(),
  ...convert(glossary),
  ...convert(read("appendix_a_evaluation_data.md")),
  ...convert(frontC, { plainH1: true, keepBold: true, skip: ["English Title Page"] }),
  new D.Paragraph({ children: [new D.PageBreak()] }),
  ...englishTitlePage,
];

const figs = CAPTIONS.filter((c) => c.kind === "fig");
const tabs = CAPTIONS.filter((c) => c.kind === "tab");

const listHeader = (word) => new D.Paragraph({
  bidirectional: true,
  alignment: D.AlignmentType.LEFT,
  spacing: { after: 160, line: LINE, lineRule: D.LineRuleType.AUTO },
  tabStops: [{ type: D.TabStopType.RIGHT, position: TOC_TAB }],
  children: [new D.TextRun({
    font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1], bold: true,
    rightToLeft: true, children: [word, new D.Tab(), "صفحه"],
  })],
});

const pageNumberFooter = new D.Footer({
  children: [new D.Paragraph({
    alignment: D.AlignmentType.CENTER,
    bidirectional: true,
    spacing: { after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
    children: [new D.TextRun({
      font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1],
      rightToLeft: true, children: [D.PageNumber.CURRENT],
    })],
  })],
});

// A section already starts on a new page, so the block that opens one
// must not also force a page break, or a blank sheet is left behind.
function section(children, pageNumbers) {
  return {
    properties: { page: { ...PAGE, ...(pageNumbers ? { pageNumbers } : {}) } },
    ...(pageNumbers ? { footers: { default: pageNumberFooter } } : {}),
    children,
  };
}

// Section one: nothing before the lists carries a page number.
const sec1 = section([
  ...titlePage,
  ...convert(frontA, { plainH1: true, keepBold: true, skip: ["صفحه عنوان فارسی"] }),
]);

// Section two: the three lists, numbered with Arabic letters.
const sec2 = section([
  plainTitle("فهرست مطالب", { pageBreakBefore: false }),
  listHeader("عنوان"),
  new D.TableOfContents("فهرست مطالب", {
    hyperlink: true, headingStyleRange: "1-3",
    hideTabAndPageNumbersInWebView: true, useAppliedParagraphOutlineLevel: true,
  }),
  plainTitle("فهرست شکل‌ها"),
  listHeader("عنوان"),
  ...figs.map(listEntry),
  plainTitle("فهرست جدول‌ها"),
  listHeader("عنوان"),
  ...tabs.map(listEntry),
], { start: 1, formatType: D.NumberFormat.ARABIC_ALPHA });

// Section three: the body, numbered from one.
const sec3 = section([...bodyBlocks, ...backBlocks],
                     { start: 1, formatType: D.NumberFormat.DECIMAL });

const doc = new D.Document({
  features: { updateFields: true },
  styles: {
    default: {
      document: {
        run: { font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1] },
        paragraph: { spacing: { line: LINE, lineRule: D.LineRuleType.AUTO },
                     bidirectional: true },
      },
      heading1: {
        run: { font: FACE, size: SZ.h1[0], sizeComplexScript: SZ.h1[1], bold: true, color: "000000" },
        paragraph: { alignment: D.AlignmentType.LEFT, bidirectional: true, outlineLevel: 0 },
      },
      heading2: {
        run: { font: FACE, size: SZ.h2[0], sizeComplexScript: SZ.h2[1], bold: true, color: "000000" },
        paragraph: { alignment: D.AlignmentType.LEFT, bidirectional: true, outlineLevel: 1 },
      },
      heading3: {
        run: { font: FACE, size: SZ.h3[0], sizeComplexScript: SZ.h3[1], bold: true, color: "000000" },
        paragraph: { alignment: D.AlignmentType.LEFT, bidirectional: true, outlineLevel: 2 },
      },
      footnoteText: {
        run: { font: FACE, size: SZ.note[0], sizeComplexScript: SZ.note[1] },
      },
    },
    paragraphStyles: [
      // The template's own caption styles, reproduced exactly.
      { id: "pic", name: "pic", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FACE, size: SZ.cap[0], sizeComplexScript: SZ.cap[1], bold: true },
        paragraph: { alignment: D.AlignmentType.CENTER, bidirectional: true,
                     spacing: { before: 60, after: 240, line: LINE, lineRule: D.LineRuleType.AUTO } } },
      { id: "table", name: "table", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FACE, size: SZ.cap[0], sizeComplexScript: SZ.cap[1], bold: true },
        paragraph: { alignment: D.AlignmentType.CENTER, bidirectional: true,
                     spacing: { before: 240, after: 60, line: LINE, lineRule: D.LineRuleType.AUTO } } },
      // The contents list, with the template's dot leader at 7926.
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal",
        run: { font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1], bold: true },
        paragraph: { bidirectional: true, spacing: { after: 100, line: LINE, lineRule: D.LineRuleType.AUTO },
                     tabStops: [{ type: D.TabStopType.RIGHT, position: TOC_TAB, leader: D.LeaderType.DOT }] } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal",
        run: { font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1] },
        paragraph: { bidirectional: true, indent: { start: 220 },
                     spacing: { after: 100, line: LINE, lineRule: D.LineRuleType.AUTO },
                     tabStops: [{ type: D.TabStopType.RIGHT, position: TOC_TAB, leader: D.LeaderType.DOT }] } },
      { id: "TOC3", name: "toc 3", basedOn: "Normal", next: "Normal",
        run: { font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1] },
        paragraph: { bidirectional: true, indent: { start: 440 },
                     spacing: { after: 100, line: LINE, lineRule: D.LineRuleType.AUTO },
                     tabStops: [{ type: D.TabStopType.RIGHT, position: TOC_TAB, leader: D.LeaderType.DOT }] } },
    ],
  },
  footnotes,
  sections: [sec1, sec2, sec3],
});

// docx-js writes neither the section-level right-to-left flags the
// template carries nor the per-page footnote restart, so both are patched
// into the package after packing. CT_SectPr has a fixed element order:
// footnotePr sits before w:type, and bidi/rtlGutter before docGrid.
function postProcess(buf) {
  const AdmZip = require("adm-zip");
  const zip = new AdmZip(buf);
  let xml = zip.readAsText("word/document.xml");

  xml = xml.replace(/<w:sectPr\b([^>]*)>((?:<w:(?:headerReference|footerReference)\b[^>]*\/>)*)/g,
    (_, attrs, refs) =>
      `<w:sectPr${attrs}>${refs}<w:footnotePr><w:numRestart w:val="eachPage"/></w:footnotePr>`);

  xml = xml.replace(/(<w:sectPr\b[^>]*>(?:(?!<\/w:sectPr>)[\s\S])*?)(<w:docGrid)/g,
    "$1<w:bidi/><w:rtlGutter/>$2");
  // An unnumbered section needs no pgNumType at all.
  xml = xml.replace(/<w:pgNumType\/>/g, "");

  zip.updateFile("word/document.xml", Buffer.from(xml, "utf8"));
  return zip.toBuffer();
}

D.Packer.toBuffer(doc).then((buf) => {
  const out = path.join(THESIS, "Antibiogram_Thesis_Haddadha.docx");
  const final = postProcess(buf);
  fs.writeFileSync(out, final);
  console.log("wrote", out,
              "| sections: 3",
              "| figures:", figs.length,
              "| tables:", tabs.length,
              "| footnotes:", footnoteSeq,
              "|", (final.length / 1024 / 1024).toFixed(2) + " MB");
});
