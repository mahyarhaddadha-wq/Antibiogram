# -*- coding: utf-8 -*-
"""Mirror the slide geometry so the deck reads right to left.

The template lays every slide out for English: the numbered circles sit to
the left of the text they label, the four method steps run left to right,
and the baseline column comes before the proposed one. Reflecting every
shape about the vertical centre line fixes all of it at once, because a
reflection preserves each shape's distance from the opposite margin.
"""
import re, glob

W = 12192000                       # slide width in EMU
SHAPE = re.compile(r"<p:(sp|pic|graphicFrame)>.*?</p:\1>", re.S)
XFRM = re.compile(r'(<a:off x=")(-?\d+)("\s+y="-?\d+"\s*/>\s*<a:ext cx=")(\d+)(")')


def mirror_block(block):
    def repl(m):
        x, cx = int(m.group(2)), int(m.group(4))
        return m.group(1) + str(W - (x + cx)) + m.group(3) + m.group(4) + m.group(5)
    return XFRM.sub(repl, block, count=1)


total = 0
for path in sorted(glob.glob("unpacked/ppt/slides/slide*.xml")):
    x = open(path, encoding="utf8").read()
    out, last, n = [], 0, 0
    for m in SHAPE.finditer(x):
        out.append(x[last:m.start()])
        block = mirror_block(m.group(0))
        n += block != m.group(0)
        out.append(block)
        last = m.end()
    out.append(x[last:])
    x = "".join(out)

    # A right arrow between two steps has to point the other way now.
    x = re.sub(r'(<a:xfrm)(>\s*<a:off[^>]*/>\s*<a:ext[^>]*/>\s*</a:xfrm>\s*<a:prstGeom prst="rightArrow")',
               r'\1 flipH="1"\2', x)

    open(path, "w", encoding="utf8").write(x)
    total += n
    print(f"  {path.split('/')[-1]:14} {n:>2} shapes mirrored")
print("total:", total)
