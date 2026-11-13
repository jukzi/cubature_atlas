"""Sort content/references.json by year and renumber every use of the reference numbers.

Usage:
    python tools/sort_references.py          # sort, renumber, rewrite the files
    python tools/sort_references.py --check  # only report whether the list is sorted

Order: by year, oldest first; entries without a year come last; entries of the same year keep their order.
The numbers n are the citation numbers [n] shown in the atlas, so they follow the order.

Places where numbers are used and rewritten:
    content/references.json                   n, and [n] inside the texts
    content/*/documentation.html              href="#rN">[N]
    rules/*/family.json                       "references": [...]
    reference_data/literature_counts.json     "refs": [...]
    site_template/index.html                  replace('[N]',cite(N))
    i18n/*.json                               [N] inside the value of the key srcNote
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def year_key(entry):
    year = str(entry.get("y", "")).strip()
    if year.isdigit():
        return (0, int(year))
    return (1, 0)


def is_sorted(refs):
    keys = [year_key(e) for e in refs]
    return keys == sorted(keys)


def build_mapping(refs):
    order = sorted(refs, key=year_key)
    mapping = {}
    for new_n, entry in enumerate(order, 1):
        mapping[entry["n"]] = new_n
    return mapping, order


def remap_list_text(text, key, mapping):
    """Rewrite the integers of every JSON array that follows the given key."""
    pattern = re.compile(r'("%s"\s*:\s*\[)([^\]]*)(\])' % re.escape(key))

    def fix(match):
        numbers = re.findall(r"\d+", match.group(2))
        if not numbers:
            return match.group(0)
        body = match.group(2)
        new_body = re.sub(r"\d+", lambda m: str(mapping[int(m.group(0))]), body)
        return match.group(1) + new_body + match.group(3)

    return pattern.sub(fix, text)


def remap_brackets(text, mapping):
    return re.sub(r"\[(\d+)\]", lambda m: "[%d]" % mapping.get(int(m.group(1)), int(m.group(1))), text)


def main():
    path = os.path.join(ROOT, "content", "references.json")
    raw = read(path)
    refs = json.loads(raw)
    if "--check" in sys.argv:
        if is_sorted(refs):
            print("OK: content/references.json is sorted by year")
            return 0
        print("ERROR: content/references.json is not sorted by year; run python tools/sort_references.py")
        return 1
    mapping, order = build_mapping(refs)
    if all(old == new for old, new in mapping.items()):
        print("already sorted, nothing to do")
        return 0
    newline = "\r\n" if "\r\n" in raw else "\n"
    out_refs = []
    for entry in order:
        entry = dict(entry)
        entry["n"] = mapping[entry["n"]]
        entry["t"] = remap_brackets(entry["t"], mapping)
        out_refs.append(entry)
    text = json.dumps(out_refs, ensure_ascii=False, indent=2).replace("\n", newline)
    if raw.endswith(newline):
        text += newline
    write(path, text)
    for doc in glob.glob(os.path.join(ROOT, "content", "*", "documentation.html")):
        html = read(doc)
        html = re.sub(r'href="#r(\d+)">\[(\d+)\]', lambda m: 'href="#r%d">[%d]' % (mapping[int(m.group(1))], mapping[int(m.group(2))]), html)
        write(doc, html)
    for fam in glob.glob(os.path.join(ROOT, "rules", "*", "family.json")):
        write(fam, remap_list_text(read(fam), "references", mapping))
    counts = os.path.join(ROOT, "reference_data", "literature_counts.json")
    write(counts, remap_list_text(read(counts), "refs", mapping))
    template = os.path.join(ROOT, "site_template", "index.html")
    html = read(template)
    html = re.sub(r"replace\('\[(\d+)\]',cite\((\d+)\)\)", lambda m: "replace('[%d]',cite(%d))" % (mapping[int(m.group(1))], mapping[int(m.group(2))]), html)
    write(template, html)
    for tr in glob.glob(os.path.join(ROOT, "i18n", "*.json")):
        src = read(tr)
        src = re.sub(r'("srcNote"\s*:\s*")((?:[^"\\]|\\.)*)(")', lambda m: m.group(1) + remap_brackets(m.group(2), mapping) + m.group(3), src)
        write(tr, src)
    print("renumbered %d references; now run: python tools/build_atlas.py --sync-hashes" % len(mapping))
    return 0


if __name__ == "__main__":
    sys.exit(main())
