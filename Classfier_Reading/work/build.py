# -*- coding: utf-8 -*-
"""Aplica REV como cambios controlados + comentarios sobre work/un/ (docx desempaquetado)."""
import copy, re, sys
from pathlib import Path
from lxml import etree

SKILL = Path(r"C:\Users\xdxdxdxd\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\3b9f5a0c-160f-44d3-8cf8-9b5cf84ca3d3\1cacdff0-3ea0-4d03-9c1c-1e3751778bde\skills\docx\scripts")
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(Path(__file__).parent))
from comment import add_comment  # noqa: E402
from revisions import REV, EXTRA  # noqa: E402

UN = Path(__file__).parent / "un"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
XML = "http://www.w3.org/XML/1998/namespace"
q = lambda t: f"{{{W}}}{t}"
AUTHOR = "Claude (propuesta IA)"
INITIALS = "IA"
DATE = "2026-09-29T12:00:00Z"
RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike",
             "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden",
             "color", "spacing", "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect",
             "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
             "specVanish", "oMath"]

_rid = [20000]
def next_id():
    _rid[0] += 1
    return str(_rid[0])

def track(tag):
    e = etree.Element(q(tag))
    e.set(q("id"), next_id()); e.set(q("author"), AUTHOR); e.set(q("date"), DATE)
    return e

def set_ordered(rpr, tag):
    """Añade <w:tag/> a rPr respetando el orden del esquema."""
    if rpr.find(q(tag)) is not None:
        return
    new = etree.Element(q(tag))
    pos = RPR_ORDER.index(tag)
    for idx, child in enumerate(rpr):
        name = etree.QName(child).localname
        if name in RPR_ORDER and RPR_ORDER.index(name) > pos:
            rpr.insert(idx, new); return
    rpr.append(new)

def base_rpr(p):
    for r in p.iter(q("r")):
        if r.find(q("t")) is not None:
            rpr = r.find(q("rPr"))
            rpr = copy.deepcopy(rpr) if rpr is not None else etree.Element(q("rPr"))
            for tag in ("i", "iCs", "b", "bCs", "rStyle"):
                for e in rpr.findall(q(tag)):
                    rpr.remove(e)
            return rpr
    return etree.Element(q("rPr"))

def make_runs(text, rpr_tpl):
    runs = []
    for k, part in enumerate(re.split(r"\*", text)):
        if not part:
            continue
        r = etree.Element(q("r"))
        rpr = copy.deepcopy(rpr_tpl)
        if k % 2 == 1:
            set_ordered(rpr, "i"); set_ordered(rpr, "iCs")
        if len(rpr):
            r.append(rpr)
        t = etree.SubElement(r, q("t")); t.text = part
        t.set(f"{{{XML}}}space", "preserve")
        runs.append(r)
    return runs

def mark_para(p, tag):
    """Marca la marca de párrafo como insertada/eliminada (w:pPr/w:rPr/w:ins|del)."""
    ppr = p.find(q("pPr"))
    if ppr is None:
        ppr = etree.Element(q("pPr")); p.insert(0, ppr)
    rpr = ppr.find(q("rPr"))
    if rpr is None:
        rpr = etree.Element(q("rPr"))
        after = [ppr.find(q(t)) for t in ("sectPr", "pPrChange")]
        after = [a for a in after if a is not None]
        if after: after[0].addprevious(rpr)
        else: ppr.append(rpr)
    rpr.insert(0, track(tag))

def delete_runs(p):
    for pe in p.findall(q("proofErr")):
        p.remove(pe)
    for r in p.findall(q("r")):
        for t in r.findall(q("t")):
            t.tag = q("delText")
        d = track("del")
        r.addprevious(d); d.append(r)

def anchor_comment(p, cid):
    start = etree.Element(q("commentRangeStart")); start.set(q("id"), str(cid))
    ppr = p.find(q("pPr"))
    if ppr is not None: ppr.addnext(start)
    else: p.insert(0, start)
    end = etree.SubElement(p, q("commentRangeEnd")); end.set(q("id"), str(cid))
    ref = etree.SubElement(p, q("r"))
    rpr = etree.SubElement(ref, q("rPr")); rs = etree.SubElement(rpr, q("rStyle")); rs.set(q("val"), "CommentReference")
    cr = etree.SubElement(ref, q("commentReference")); cr.set(q("id"), str(cid))

def comment(p, text):
    cid, _, _ = add_comment(UN, text, author=AUTHOR, initials=INITIALS)
    anchor_comment(p, cid)

def main():
    path = UN / "word" / "document.xml"
    tree = etree.parse(str(path))
    body = tree.getroot().find(q("body"))
    ps = [e for e in body if e.tag == q("p")]
    assert len(ps) == 586, len(ps)

    # Comentario general primero, para que quede como id 0.
    comment(ps[0], EXTRA[0])

    for idx in sorted(REV):
        new, note = REV[idx]
        p = ps[idx]
        orig = "".join(p.itertext())
        lead = orig[: len(orig) - len(orig.lstrip())]
        rpr_tpl = base_rpr(p)
        delete_runs(p)
        if new is None:
            mark_para(p, "del")
        else:
            chunks = new.split("\n")
            ins = track("ins")
            for r in make_runs(lead + chunks[0], rpr_tpl):
                ins.append(r)
            p.append(ins)
            prev = p
            for k, chunk in enumerate(chunks[1:], start=1):
                mark_para(prev, "ins")  # la marca que separa este trozo del anterior es nueva
                np_ = etree.Element(q("p"))
                ppr = p.find(q("pPr"))
                if ppr is not None:
                    ppr2 = copy.deepcopy(ppr)
                    rpr2 = ppr2.find(q("rPr"))
                    if rpr2 is not None:
                        for tr in rpr2.findall(q("ins")) + rpr2.findall(q("del")):
                            rpr2.remove(tr)
                    np_.append(ppr2)
                ins2 = track("ins")
                for r in make_runs(chunk, rpr_tpl):
                    ins2.append(r)
                np_.append(ins2)
                prev.addnext(np_)
                prev = np_
        if note:
            comment(p, note)

    for idx, note in sorted(EXTRA.items()):
        if idx != 0:
            comment(ps[idx], note)

    # Quitar paraId/textId duplicados en párrafos clonados
    seen = set()
    for p in body.iter(q("p")):
        pid = p.get(f"{{{W14}}}paraId")
        if pid in seen:
            del p.attrib[f"{{{W14}}}paraId"]
            p.attrib.pop(f"{{{W14}}}textId", None)
        elif pid:
            seen.add(pid)

    tree.write(str(path), xml_declaration=True, encoding="UTF-8", standalone=True)
    print("OK: cambios aplicados a", len(REV), "párrafos; comentarios:", sum(1 for v in REV.values() if v[1]) + len(EXTRA))

if __name__ == "__main__":
    main()
