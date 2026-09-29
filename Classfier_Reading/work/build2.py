# -*- coding: utf-8 -*-
"""Ronda 2: ediciones puntuales con control de cambios + respuestas a comentarios sobre work/un2/."""
import copy, re, sys
from pathlib import Path
from lxml import etree

SKILL = Path(r"C:\Users\xdxdxdxd\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\3b9f5a0c-160f-44d3-8cf8-9b5cf84ca3d3\1cacdff0-3ea0-4d03-9c1c-1e3751778bde\skills\docx\scripts")
sys.path.insert(0, str(SKILL))
from comment import add_comment  # noqa: E402

UN = Path(__file__).parent / "un2"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
q = lambda t: f"{{{W}}}{t}"
AUTHOR, INITIALS, DATE = "Claude (propuesta IA)", "IA", "2026-09-29T18:00:00Z"
_rid = [30000]

def track(tag):
    _rid[0] += 1
    e = etree.Element(q(tag))
    e.set(q("id"), str(_rid[0])); e.set(q("author"), AUTHOR); e.set(q("date"), DATE)
    return e

def text_runs(p):
    out = []
    for r in p.findall(q("r")):
        ts = r.findall(q("t"))
        if ts:
            out.append((r, "".join(t.text or "" for t in ts)))
    return out

def simple(r):
    return all(etree.QName(c).localname in ("rPr", "t") for c in r) and len(r.findall(q("t"))) == 1

def split_run(r, k):
    """Parte un run simple en dos en la posición k; devuelve (izq, der)."""
    assert simple(r), "run complejo en un límite de edición"
    t = r.find(q("t")); s = t.text or ""
    r2 = copy.deepcopy(r)
    t.text = s[:k]; t.set(f"{{{XML}}}space", "preserve")
    t2 = r2.find(q("t")); t2.text = s[k:]; t2.set(f"{{{XML}}}space", "preserve")
    r.addnext(r2)
    return r, r2

def boundary(p, pos):
    """Garantiza que exista un límite de run en la posición pos del texto; devuelve el run que empieza en pos (o None si es el final)."""
    acc = 0
    for r, s in text_runs(p):
        if acc == pos:
            return r
        if acc < pos < acc + len(s):
            _, right = split_run(r, pos - acc)
            return right
        acc += len(s)
    return None

def new_runs(text, rpr_src):
    runs = []
    for k, part in enumerate(re.split(r"\*", text)):
        if not part:
            continue
        r = etree.Element(q("r"))
        if rpr_src is not None:
            rpr = copy.deepcopy(rpr_src)
            for tag in ("i", "iCs"):
                for e in rpr.findall(q(tag)):
                    rpr.remove(e)
            if k % 2 == 1:
                i_el = etree.Element(q("i")); ic = etree.Element(q("iCs"))
                # después de rFonts/b/bCs si existen
                anchor = None
                for tag in ("bCs", "b", "rFonts", "rStyle"):
                    anchor = rpr.find(q(tag))
                    if anchor is not None:
                        break
                if anchor is not None:
                    anchor.addnext(i_el); i_el.addnext(ic)
                else:
                    rpr.insert(0, ic); rpr.insert(0, i_el)
            r.append(rpr)
        t = etree.SubElement(r, q("t")); t.text = part; t.set(f"{{{XML}}}space", "preserve")
        runs.append(r)
    return runs

def sub(p, old, new):
    for pe in p.findall(q("proofErr")):
        p.remove(pe)
    full = "".join(s for _, s in text_runs(p))
    start = full.find(old)
    assert start >= 0, f"no encontrado: {old[:60]!r} en {full[:80]!r}"
    end = start + len(old)
    if old:
        first = boundary(p, start)
        after = boundary(p, end)
        # runs dentro de [start, end)
        acc, dels = 0, []
        for r, s in text_runs(p):
            if start <= acc < end:
                dels.append(r)
            acc += len(s)
        rpr_src = dels[0].find(q("rPr"))
        last = None
        for r in dels:
            for t in r.findall(q("t")):
                t.tag = q("delText")
            d = track("del"); r.addprevious(d); d.append(r); last = d
        anchor = last
    else:
        raise ValueError("usar ins_after para inserciones")
    if new:
        ins = track("ins")
        for r in new_runs(new, rpr_src):
            ins.append(r)
        anchor.addnext(ins)

def ins_after(p, anchor_text, new):
    for pe in p.findall(q("proofErr")):
        p.remove(pe)
    full = "".join(s for _, s in text_runs(p))
    pos = full.find(anchor_text)
    assert pos >= 0, f"ancla no encontrada: {anchor_text!r}"
    pos += len(anchor_text)
    nxt = boundary(p, pos)
    prev_run = None
    acc = 0
    for r, s in text_runs(p):
        if acc + len(s) == pos:
            prev_run = r
        acc += len(s)
    ins = track("ins")
    for r in new_runs(new, prev_run.find(q("rPr"))):
        ins.append(r)
    prev_run.addnext(ins)

def mark_para(p, tag):
    ppr = p.find(q("pPr"))
    if ppr is None:
        ppr = etree.Element(q("pPr")); p.insert(0, ppr)
    rpr = ppr.find(q("rPr"))
    if rpr is None:
        rpr = etree.Element(q("rPr"))
        tail = [ppr.find(q(t)) for t in ("sectPr", "pPrChange")]
        tail = [x for x in tail if x is not None]
        (tail[0].addprevious(rpr) if tail else ppr.append(rpr))
    rpr.insert(0, track(tag))

def delete_para(p):
    for pe in p.findall(q("proofErr")):
        p.remove(pe)
    for r in p.findall(q("r")):
        for t in r.findall(q("t")):
            t.tag = q("delText")
        d = track("del"); r.addprevious(d); d.append(r)
    mark_para(p, "del")

def add_para_after(p_ref, segments):
    """Inserta un párrafo nuevo (marcado como insertado) tras p_ref con su mismo pPr."""
    np_ = etree.Element(q("p"))
    ppr = copy.deepcopy(p_ref.find(q("pPr")))
    np_.append(ppr)
    runs_src = [r for r, _ in text_runs(p_ref)]
    rpr_src = runs_src[0].find(q("rPr")) if runs_src else None
    ins = track("ins")
    for r in new_runs(segments, rpr_src):
        ins.append(r)
    np_.append(ins)
    p_ref.addnext(np_)
    mark_para(p_ref, "ins")  # la marca que separa p_ref del nuevo párrafo es nueva
    return np_

def anchor_new(p, cid):
    s = etree.Element(q("commentRangeStart")); s.set(q("id"), str(cid))
    ppr = p.find(q("pPr"))
    (ppr.addnext(s) if ppr is not None else p.insert(0, s))
    e = etree.SubElement(p, q("commentRangeEnd")); e.set(q("id"), str(cid))
    p.append(ref_run(cid))

def ref_run(cid):
    r = etree.Element(q("r"))
    rpr = etree.SubElement(r, q("rPr")); rs = etree.SubElement(rpr, q("rStyle")); rs.set(q("val"), "CommentReference")
    c = etree.SubElement(r, q("commentReference")); c.set(q("id"), str(cid))
    return r

def reply(body, parent_id, text):
    cid, _, _ = add_comment(UN, text, author=AUTHOR, initials=INITIALS, parent_id=parent_id)
    pid = str(parent_id)
    st = next(e for e in body.iter(q("commentRangeStart")) if e.get(q("id")) == pid)
    en = next(e for e in body.iter(q("commentRangeEnd")) if e.get(q("id")) == pid)
    rf = next(e for e in body.iter(q("commentReference")) if e.get(q("id")) == pid)
    s = etree.Element(q("commentRangeStart")); s.set(q("id"), str(cid)); st.addnext(s)
    e = etree.Element(q("commentRangeEnd")); e.set(q("id"), str(cid)); en.addprevious(e)
    rf.getparent().addnext(ref_run(cid))

def main():
    from edits2 import EDITS, DELETE, NEW_REFS, REPLIES, NEW_COMMENTS
    path = UN / "word" / "document.xml"
    tree = etree.parse(str(path))
    body = tree.getroot().find(q("body"))
    ps = [e for e in body if e.tag == q("p")]
    assert len(ps) == 584, len(ps)

    for idx, kind, a, b in EDITS:
        if kind == "sub":
            sub(ps[idx], a, b)
        elif kind == "ins":
            ins_after(ps[idx], a, b)
    for idx in DELETE:
        delete_para(ps[idx])
    prev = ps[NEW_REFS[0]]
    for seg in NEW_REFS[1]:
        prev = add_para_after(prev, seg)
    for pid, text in REPLIES:
        reply(body, pid, text)
    for idx, text in NEW_COMMENTS:
        cid, _, _ = add_comment(UN, text, author=AUTHOR, initials=INITIALS)
        anchor_new(ps[idx], cid)
    tree.write(str(path), xml_declaration=True, encoding="UTF-8", standalone=True)
    print("OK:", len(EDITS), "ediciones,", len(DELETE), "párrafos eliminados,", len(NEW_REFS[1]), "referencias nuevas,",
          len(REPLIES), "respuestas,", len(NEW_COMMENTS), "comentarios nuevos")

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
