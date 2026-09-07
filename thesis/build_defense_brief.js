#!/usr/bin/env node
/**
 * Build the defence content brief: what belongs in the presentation, what
 * to say on each slide, every number that may be asked about, and the
 * questions the committee is likely to put.
 *
 * Everything here is taken from the thesis as it now stands, after the
 * author's revision. Set in B Nazanin, to match the deck.
 *
 *   node thesis/build_defense_brief.js
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");

const FA = "B Nazanin";
const EN = "Times New Roman";
const FACE = { ascii: EN, hAnsi: EN, eastAsia: EN, cs: FA };
const LINE = 276;
const W = 11906 - 1701 * 2;

const t = (text, o = {}) => new D.TextRun({
  text, font: FACE, size: o.size || 24, sizeComplexScript: o.sizeCs || 26,
  bold: !!o.bold, color: o.color, italics: !!o.italics, rightToLeft: o.rtl !== false,
});

const P = (text, o = {}) => new D.Paragraph({
  children: Array.isArray(text) ? text : [t(text, o)],
  bidirectional: true,
  alignment: o.align || D.AlignmentType.BOTH,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: LINE, lineRule: D.LineRuleType.AUTO },
  indent: o.indent,
  ...(o.border ? { border: o.border } : {}),
});

const H = (text, lvl) => new D.Paragraph({
  children: [t(text, { bold: true, size: lvl === 1 ? 32 : 28, sizeCs: lvl === 1 ? 34 : 30,
                       color: lvl === 1 ? "8C1515" : "1F3864" })],
  heading: lvl === 1 ? D.HeadingLevel.HEADING_1 : D.HeadingLevel.HEADING_2,
  bidirectional: true,
  alignment: D.AlignmentType.LEFT,
  keepNext: true,
  spacing: { before: lvl === 1 ? 360 : 260, after: 140, line: LINE, lineRule: D.LineRuleType.AUTO },
  ...(lvl === 1 ? { pageBreakBefore: true } : {}),
});

const bullet = (text) => P(text, { indent: { left: 340, hanging: 180 }, after: 70 });

// A boxed "what to say" block, so the spoken text is never confused with
// the notes about it.
const say = (lines) => new D.Table({
  columnWidths: [W],
  width: { size: W, type: D.WidthType.DXA },
  visuallyRightToLeft: true,
  rows: [new D.TableRow({
    children: [new D.TableCell({
      width: { size: W, type: D.WidthType.DXA },
      shading: { type: D.ShadingType.CLEAR, fill: "F4F6F9" },
      margins: { top: 120, bottom: 120, left: 160, right: 160 },
      borders: {
        top: { style: D.BorderStyle.NONE }, bottom: { style: D.BorderStyle.NONE },
        left: { style: D.BorderStyle.NONE },
        right: { style: D.BorderStyle.SINGLE, size: 18, color: "8C1515" },
      },
      children: lines.map((l, i) => P(l, { after: i === lines.length - 1 ? 0 : 100 })),
    })],
  })],
});

function table(rows, widths) {
  const cw = widths || Array(rows[0].length).fill(Math.floor(W / rows[0].length));
  const mk = (cells, head) => new D.TableRow({
    tableHeader: !!head, cantSplit: true,
    children: cells.map((c, i) => new D.TableCell({
      width: { size: cw[i], type: D.WidthType.DXA },
      shading: head ? { type: D.ShadingType.CLEAR, fill: "E7ECF4" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      verticalAlign: D.VerticalAlign.CENTER,
      children: [P(c, { align: D.AlignmentType.CENTER, after: 0,
                        size: 22, sizeCs: 24, bold: !!head })],
    })),
  });
  return new D.Table({
    columnWidths: cw, width: { size: W, type: D.WidthType.DXA },
    visuallyRightToLeft: true,
    rows: [mk(rows[0], true), ...rows.slice(1).map((r) => mk(r))],
  });
}

const gap = (after = 200) => new D.Paragraph({ text: "", spacing: { after } });

module.exports = { D, FA, EN, FACE, LINE, W, t, P, H, bullet, say, table, gap };
