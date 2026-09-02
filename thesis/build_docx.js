#!/usr/bin/env node
/**
 * Build the final Persian thesis .docx from the markdown chapters.
 *
 * Layout follows the faculty writing guide: A4, right to left, B Nazanin
 * 14 for Persian and Times New Roman 12 for Latin, B Titr for headings,
 * 2.5 cm margins with a 0.5 cm gutter, single line spacing, table caption
 * above and figure caption below, footnote numbering restarting on each
 * page, and an IEEE reference list set left to right in English.
 *
 * Citation and footnote markers sit in right-to-left text and therefore
 * render with Persian digits; the reference entries and the footnote
 * bodies are left-to-right English and render with Latin digits.
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
const FA = "B Nazanin";        // Persian body face
const FA_HEAD = "B Titr";      // Persian heading face
const EN = "Times New Roman";  // Latin face

const FACE = { ascii: EN, hAnsi: EN, eastAsia: EN, cs: FA };
const FACE_HEAD = { ascii: EN, hAnsi: EN, eastAsia: EN, cs: FA_HEAD };

// Latin size / complex-script size, in half points.
const SZ = {
  body: [24, 28], h1: [36, 36], h2: [26, 28], h3: [22, 26],
  cap: [20, 24], tbl: [20, 24], note: [20, 20], code: [18, 18],
};

const CONTENT_W = 11907 - 1418 * 2 - 284; // usable width in DXA

const PAGE = {
  size: { width: 11907, height: 16839 },
  margin: { top: 1418, right: 1418, bottom: 1134, left: 1418,
            header: 720, footer: 567, gutter: 284 },
};

const LINE = 240;            // single line spacing
const FIRST_LINE = 288;      // 0.5 cm first-line indent

const toFa = (n) => String(n).replace(/\d/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]);
const isLatin = (t) => !/[\u0600-\u06FF]/.test(t);

// ─── Footnote state ────────────────────────────────────────────────────
// Markdown carries per-file keys ([^1], [^2] ...); the document needs one
// continuous sequence, so keys are remapped as each file is converted.
// Word renumbers the printed marks per page by itself.
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
      spacing: { after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
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
function runs(text, opt = {}) {
  const [sz, szCs] = opt.size || SZ.body;
  const face = opt.face || FACE;
  const rtl = opt.rtl !== false;
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
    } else if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) {
      push(tok.slice(1, -1), { font: { ascii: "Consolas", hAnsi: "Consolas", cs: "Consolas" },
                               size: SZ.code[0], sizeComplexScript: SZ.code[1], rightToLeft: false });
    } else if (tok.startsWith("$")) {
      push(tok.slice(1, -1), { italics: true, rightToLeft: false });
    } else push(tok.slice(1, -1), { italics: true });
    last = re.lastIndex;
  }
  push(text.slice(last));
  return out.length ? out : [new D.TextRun({ ...base, text: "" })];
}

// A body paragraph: right to left, justified, single spaced, first line
// indented, exactly as the NewParagraph style of the writing guide.
function para(text, o = {}) {
  const latin = isLatin(text.replace(/[*`$\[\]^]/g, "").trim());
  return new D.Paragraph({
    children: runs(text, { ...o, rtl: !latin }),
    bidirectional: !latin,
    alignment: o.alignment || (latin ? D.AlignmentType.LEFT : D.AlignmentType.BOTH),
    spacing: { before: o.before ?? 120, after: o.after ?? 0,
               line: LINE, lineRule: D.LineRuleType.AUTO },
    indent: o.indent !== undefined ? o.indent : { firstLine: FIRST_LINE },
    ...(o.border ? { border: o.border } : {}),
  });
}

const blank = (after = 0) => new D.Paragraph({
  text: "", spacing: { after, line: LINE, lineRule: D.LineRuleType.AUTO },
});

// ─── Headings ──────────────────────────────────────────────────────────
// In a right-to-left paragraph, w:jc="left" is the leading edge, so these
// sit against the right margin, as the faculty template has them.
function heading(text, level) {
  const cfg = {
    1: { size: SZ.h1, heading: D.HeadingLevel.HEADING_1, before: 240, after: 240 },
    2: { size: SZ.h2, heading: D.HeadingLevel.HEADING_2, before: 360, after: 60 },
    3: { size: SZ.h3, heading: D.HeadingLevel.HEADING_3, before: 240, after: 60 },
  }[level];
  return new D.Paragraph({
    children: [new D.TextRun({
      font: FACE_HEAD, size: cfg.size[0], sizeComplexScript: cfg.size[1],
      bold: true, rightToLeft: true, text,
    })],
    heading: cfg.heading,
    bidirectional: true,
    alignment: D.AlignmentType.LEFT,
    keepNext: true,
    spacing: { before: cfg.before, after: cfg.after, line: LINE, lineRule: D.LineRuleType.AUTO },
    ...(level === 1 ? { pageBreakBefore: true } : {}),
  });
}

const centeredHeading = (text) => new D.Paragraph({
  children: [new D.TextRun({
    font: FACE_HEAD, size: SZ.h2[0], sizeComplexScript: SZ.h2[1],
    bold: true, rightToLeft: true, text,
  })],
  heading: D.HeadingLevel.HEADING_1,
  bidirectional: true,
  alignment: D.AlignmentType.CENTER,
  pageBreakBefore: true,
  spacing: { before: 240, after: 240, line: LINE, lineRule: D.LineRuleType.AUTO },
});

// ─── Captions ──────────────────────────────────────────────────────────
const caption = (text, above) => new D.Paragraph({
  children: runs(text, { size: SZ.cap, run: { bold: true } }),
  bidirectional: true,
  alignment: D.AlignmentType.CENTER,
  keepNext: !!above,
  spacing: above
    ? { before: 240, after: 60, line: LINE, lineRule: D.LineRuleType.AUTO }
    : { before: 60, after: 240, line: LINE, lineRule: D.LineRuleType.AUTO },
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
    spacing: { before: 360, after: 120, line: LINE, lineRule: D.LineRuleType.AUTO },
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
      if (depth === 1) out.push(opts.centeredH1 ? centeredHeading(text) : heading(text, 1));
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
        spacing: { after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
        bidirectional: false,
      })));
      out.push(blank(120));
      continue;
    }

    // Table caption, kept above its table as the writing guide requires
    if ((m = t.match(/^\*\*جدول\s*([۰-۹]+-[۰-۹]+)\s*[:：](.*?)\*\*$/))) {
      pendingCaption = caption(`جدول ${m[1]}: ${m[2].trim()}`, true);
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

    if (/^\$\$.*\$\$$/.test(t)) {
      out.push(new D.Paragraph({
        children: [new D.TextRun({
          text: t.slice(2, -2).trim(), font: FACE,
          size: SZ.body[0], sizeComplexScript: SZ.body[1],
          italics: true, rightToLeft: false,
        })],
        alignment: D.AlignmentType.CENTER,
        bidirectional: false,
        spacing: { before: 140, after: 160, line: LINE, lineRule: D.LineRuleType.AUTO },
      }));
      i++; continue;
    }

    // Figure caption, kept below its figure as the writing guide requires
    if ((m = t.match(/^\*{1,2}(?:شکلِ?|شکل)\s*([۰-۹]+-[۰-۹]+)\s*[:：](.*)$/))) {
      const num = m[1];
      const capText = `شکل ${num}: ${m[2].replace(/\*+$/, "").trim()}`;
      const f = FIGURES[num];
      if (f && fs.existsSync(f)) out.push(image(f, num.startsWith("۴") ? 320 : 420));
      else console.warn("  missing figure", num, f);
      out.push(caption(capText, false));
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
        before: 60,
      })));
      out.push(blank(120));
      continue;
    }

    if (/^[-*]\s+/.test(t) || /^\d+[.)]\s+/.test(t) || /^[۰-۹]+[.)]\s+/.test(t)) {
      const bullet = /^[-*]\s+/.test(t);
      out.push(para(t.replace(/^[-*]\s+/, "• ").replace(/^([۰-۹\d]+[.)])\s+/, "$1 "), {
        indent: { left: 340, hanging: 180 },
        before: 40,
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
    out.push(para(buf.join(" ")));
  }
  if (pendingCaption) out.push(pendingCaption);
  return out;
}

// ─── Document assembly ─────────────────────────────────────────────────
const read = (f) => fs.readFileSync(path.join(THESIS, f), "utf8");

const centered = (text, size, bold = true, after = 160, en = false) => new D.Paragraph({
  children: [new D.TextRun({
    font: en ? FACE : FACE_HEAD, size: size[0], sizeComplexScript: size[1],
    bold, text, rightToLeft: !en,
  })],
  alignment: D.AlignmentType.CENTER,
  bidirectional: !en,
  spacing: { after, line: LINE, lineRule: D.LineRuleType.AUTO },
});

// ---- References: left to right, English, hanging indent, Latin numbers
function referenceParagraphs() {
  const md = read("references.md");
  const rows = [...md.matchAll(/^\|\s*\[(\d+)\]\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$/gm)];
  const out = [para(
    "ارجاع‌دهی در این نوشتار بر پایه سبک IEEE و به ترتیب نخستین ظهور در متن انجام شده است. " +
    "شماره ارجاع در متن فارسی و درون کروشه فارسی آمده است، و مشخصات هر منبع در همین فهرست به " +
    "انگلیسی و درون کروشه انگلیسی نوشته شده است. اطلاعات کتاب‌شناختی منابع نمایه‌شده در PubMed " +
    "با جستجوی مستقیم عنوان در آن پایگاه راستی‌آزمایی شده و شناسه PMID و DOI هر مورد ثبت گردیده " +
    "است، تا هر ارجاع مستقلاً قابل بررسی بماند."
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
        size: SZ.note[0], sizeComplexScript: SZ.note[1], rightToLeft: false,
      })],
    }));
  });
  return out;
}

// ---- Title page (Persian) ----
const titlePage = [
  blank(600),
  centered("دانشگاه اصفهان", SZ.h1),
  centered("دانشکده فنی و مهندسی", SZ.h2),
  centered("گروه مهندسی پزشکی", SZ.h3, true, 900),
  centered("پروژه کارشناسی رشته مهندسی پزشکی", SZ.h1, true, 700),
  centered("طراحی و پیاده‌سازی سامانه هوشمند تحلیل خودکار آزمون آنتی‌بیوگرام", SZ.h2, true, 90),
  centered("مبتنی بر پردازش تصویر و یادگیری ماشین مطابق استاندارد EUCAST", SZ.h2, true, 900),
  centered("استاد راهنما:", SZ.h2, true, 90),
  centered("دکتر محمدرضا یزدچی", SZ.h2, true, 600),
  centered("دانشجو:", SZ.h2, true, 90),
  centered("مهیار حدادها", SZ.h2, true, 60),
  centered("۴۰۱۲۰۱۳۰۵۳", SZ.h3, true, 900),
  centered("شهریور ۱۴۰۵", SZ.h3),
];

const englishTitlePage = [
  blank(600),
  centered("University of Isfahan", SZ.h1, true, 140, true),
  centered("Faculty of Engineering", SZ.h2, true, 100, true),
  centered("Department of Biomedical Engineering", SZ.h3, true, 900, true),
  centered("B.Sc. Project", SZ.h1, true, 700, true),
  centered("Design and Implementation of an Intelligent Automated", SZ.h2, true, 60, true),
  centered("Antibiogram Analysis System Based on Image Processing", SZ.h2, true, 60, true),
  centered("and Machine Learning According to EUCAST Standards", SZ.h2, true, 900, true),
  centered("Supervisor:", SZ.h2, true, 90, true),
  centered("Dr. Mohammad Reza Yazdchi", SZ.h2, true, 600, true),
  centered("By:", SZ.h2, true, 90, true),
  centered("Mahyar Haddadha", SZ.h2, true, 900, true),
  centered("September 2026", SZ.h3, true, 160, true),
];

// ---- Front matter is split so page numbering can follow the guide ----
// Nothing before the lists is numbered, the lists carry Abjad letters, and
// the body is numbered from one. The English abstract closes the volume.
const front = read("front_matter.md");
const cutLists = front.indexOf("# فهرست مطالب");
const cutEnglish = front.indexOf("# Abstract");
const frontA = front.slice(0, cutLists);              // certificate, thanks, abstract
const frontB = front.slice(cutLists, cutEnglish);     // lists and glossary
const frontC = front.slice(cutEnglish);               // English abstract

const pageNumberFooter = new D.Footer({
  children: [new D.Paragraph({
    alignment: D.AlignmentType.CENTER,
    bidirectional: true,
    spacing: { after: 0, line: LINE, lineRule: D.LineRuleType.AUTO },
    children: [new D.TextRun({
      font: FACE, size: SZ.note[0], sizeComplexScript: SZ.note[1],
      rightToLeft: true, children: [D.PageNumber.CURRENT],
    })],
  })],
});

function section(children, pageNumbers) {
  return {
    properties: { page: { ...PAGE, ...(pageNumbers ? { pageNumbers } : {}) } },
    ...(pageNumbers ? { footers: { default: pageNumberFooter } } : {}),
    children,
  };
}

// Section 1: nothing before the lists is numbered.
const sec1 = section([
  ...titlePage,
  ...convert(frontA, { centeredH1: true, skip: ["صفحه عنوان فارسی"] }),
]);

// Section 2: the lists carry Abjad letters.
const sec2 = section([
  centeredHeading("فهرست مطالب"),
  new D.TableOfContents("فهرست مطالب", {
    hyperlink: true, headingStyleRange: "1-3", rightTabStop: CONTENT_W,
  }),
  ...convert(frontB, { centeredH1: true, skip: ["فهرست مطالب"] }),
], { start: 1, formatType: D.NumberFormat.ARABIC_ABJAD });

// Section 3: the body, numbered from one.
const sec3 = section([
  ...convert(read("chapter_01.md")),
  ...convert(read("chapter_02.md")),
  ...convert(read("chapter_03.md")),
  ...convert(read("chapter_04.md")),
  ...convert(read("chapter_05.md")),
  ...convert(read("chapter_06.md")),
  // The writing guide puts the appendices before the reference list.
  ...convert(read("appendix_a_evaluation_data.md"), { centeredH1: true }),
  centeredHeading("منابع و مآخذ"),
  ...referenceParagraphs(),
  ...convert(frontC, { centeredH1: true, skip: ["English Title Page"] }),
  new D.Paragraph({ children: [new D.PageBreak()] }),
  ...englishTitlePage,
], { start: 1, formatType: D.NumberFormat.DECIMAL });

const doc = new D.Document({
  styles: {
    default: {
      document: {
        run: { font: FACE, size: SZ.body[0], sizeComplexScript: SZ.body[1] },
        paragraph: { spacing: { line: LINE, lineRule: D.LineRuleType.AUTO } },
      },
    },
  },
  footnotes,
  sections: [sec1, sec2, sec3],
});

// docx-js writes neither the section-level right-to-left flags that the
// faculty template carries nor the per-page footnote restart, so both are
// patched into the package after packing. CT_SectPr has a fixed element
// order: footnotePr sits before w:type, and bidi/rtlGutter before docGrid.
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
              "| footnotes:", footnoteSeq,
              "|", (final.length / 1024 / 1024).toFixed(2) + " MB");
});
