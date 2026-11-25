"""Frontend syntax, i18n integrity, and 404.html router tests."""

import json
import os
import re
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_atlas


class FrontendTests(unittest.TestCase):
    def test_template_scripts_parse_cleanly_in_node(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        with open(os.path.join(ROOT, "site_template", "index.html"), encoding="utf8") as fh:
            tpl = fh.read()
        scripts = re.findall(r"<script>([\s\S]*?)</script>", tpl)
        self.assertGreaterEqual(len(scripts), 1)
        for idx, body in enumerate(scripts):
            js = re.sub(r"__[A-Z0-9_]+__", "null", body)
            proc = subprocess.run(
                [node, "-e", "new Function(require('fs').readFileSync(0,'utf8'))"],
                input=js, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 0, f"Script #{idx} syntax error:\n{proc.stderr}")

    def test_template_data_i_keys_exist_in_en_and_de(self):
        with open(os.path.join(ROOT, "site_template", "index.html"), encoding="utf8") as fh:
            tpl = fh.read()
        with open(os.path.join(ROOT, "i18n", "en.json"), encoding="utf8") as fh:
            en = json.load(fh)
        with open(os.path.join(ROOT, "i18n", "de.json"), encoding="utf8") as fh:
            de = json.load(fh)
        keys = set(re.findall(r'data-i="([^"]+)"', tpl))
        self.assertTrue(keys)
        for k in sorted(keys):
            self.assertIn(k, en, f"data-i='{k}' missing from i18n/en.json")
            self.assertIn(k, de, f"data-i='{k}' missing from i18n/de.json")

    def test_404_router_handles_en_de_preview_and_invalid_paths(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        fams = [
            {"id": "xiao_gimbutas_2010", "role": "literature", "rules": [{"domain": "triangle"}]},
            {"id": "minimal_pi", "role": "pi", "rules": [{"domain": "tetra"}]},
        ]
        page = build_atlas.render_404(fams)
        m = re.search(r"<script>([\s\S]*?)</script>", page)
        self.assertIsNotNone(m)
        script = m.group(1)
        harness = """
const script = process.argv[1];
const cases = [
  ["/cubature_atlas/tetra/npi/8/", "", "", "/cubature_atlas/?dom=tetra&src=found&n=8"],
  ["/cubature_atlas/de/square/npo/12/", "?mode=rel", "#tbl", "/cubature_atlas/de/?dom=square&src=out&n=12&mode=rel#tbl"],
  ["/cubature_atlas/pr-preview/pr-412/cube/npi/53/", "", "", "/cubature_atlas/pr-preview/pr-412/?dom=cube&src=found&n=53"],
  ["/cubature_atlas/pr-preview/pr-412/de/triangle/xiao_gimbutas_2010/10/", "", "", "/cubature_atlas/pr-preview/pr-412/de/?dom=triangle&src=f%3Axiao_gimbutas_2010&n=10"],
  ["/square/", "", "", "/?dom=square"],
  ["/cubature_atlas/interval/lob/5/", "", "", "/cubature_atlas/?dom=interval&src=lob&n=5"],
  ["/cubature_atlas/tetra/invalid_src/8/", "", "", null],
  ["/cubature_atlas/rules/foo/square/", "", "", null],
  ["/cubature_atlas/not-a-domain/", "", "", null],
];
for (const [pathname, search, hash, expected] of cases) {
  let replaced = null;
  const location = { pathname, search, hash, replace(u) { replaced = u; } };
  new Function("location", script)(location);
  if (replaced !== expected) {
    throw new Error(`For ${pathname}${search}${hash}: got ${replaced}, expected ${expected}`);
  }
}
"""
        proc = subprocess.run([node, "-e", harness, script], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, f"404 router test failed:\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
