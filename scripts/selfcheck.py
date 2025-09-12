import sys, re, json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
templates = list((ROOT/"templates").glob("*.html"))
errors = []

# 1) Duplicate scripts
for tpl in templates:
    html = tpl.read_text(encoding="utf-8", errors="ignore")
    srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
    dups = [s for i,s in enumerate(srcs) if s in srcs[:i]]
    if dups:
        errors.append(f"{tpl.name}: duplicate <script> {set(dups)}")

# 2) JS redeclare basics: look for common globals declared multiple times across our new files
jsdir = ROOT/"static/js"
globals = {}
for js in jsdir.glob("*.js"):
    txt = js.read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r'if\(!window\.([a-zA-Z0-9_]+)\)\{', txt)
    for g in m:
        globals.setdefault(g, []).append(js.name)
for g, files in globals.items():
    if len(files) > 1 and g not in {"guards"}:
        errors.append(f"Global {g} declared in multiple files: {files}")

# 3) i18n coverage
ru = json.loads((ROOT/"static/i18n/ru.json").read_text(encoding="utf-8"))
en = json.loads((ROOT/"static/i18n/en.json").read_text(encoding="utf-8"))
keys = set(ru.keys()) | set(en.keys())
for tpl in templates:
    html = tpl.read_text(encoding="utf-8", errors="ignore")
    for key in re.findall(r'data-i18n="([^"]+)"', html):
        if key not in ru or key not in en:
            errors.append(f"Missing i18n key '{key}' in ru/en")

# 4) Quick syntax scan: no 'Unexpected identifier' patterns
app = (ROOT/"app.py").read_text(encoding="utf-8", errors="ignore")
if "https://fonts.gstatic.com\"lf" in app:
    errors.append("CSP typo detected: fonts.gstatic.com\\\"lf")

print("OK" if not errors else "FAIL")
for e in errors: print("-", e)
