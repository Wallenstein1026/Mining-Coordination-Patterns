"""Check release structure without importing ML packages or claiming data validity."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/'src').rglob('*.py'):compile(path.read_text(),str(path.relative_to(ROOT)),'exec')
for path in ROOT.rglob('*.md'):
    if any(p in {'.venv','.git','__pycache__'} for p in path.relative_to(ROOT).parts):continue
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if target.startswith(('https://','http://','#','mailto:')):continue
        assert (path.parent/target.split('#',1)[0]).exists(),f'Broken link: {path.name}: {target}'
assert json.loads((ROOT/'PROJECT_STATUS.json').read_text())['group']==28
assert not (ROOT/'archive').exists() and not (ROOT/'README_EXPLORATORY.md').exists()
assert not (ROOT/'Poster.pdf').exists() and not (ROOT/'Report.pdf').exists()
for directory in ['results','figures']:
    assert (ROOT/directory/'README.md').exists()
print('Source syntax, local documentation links and current release structure passed.')
print('No dataset download, official experiment, or scientific claim is validated by this check.')
