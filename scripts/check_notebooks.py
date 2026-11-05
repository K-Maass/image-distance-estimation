"""Reject execution outputs and identity metadata in publishable notebooks."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', '*.ipynb'], cwd=root
    ).decode().split('\0')
    errors = []
    for name in sorted(set(filter(None, names))):
        path = root / name
        if not path.exists():
            continue  # Deleted in the release tree.
        notebook = json.loads(path.read_text(encoding='utf-8'))
        if set(notebook.get('metadata', {})) - {'kernelspec', 'language_info'}:
            errors.append(f'{name}: unexpected notebook metadata')
        for i, cell in enumerate(notebook.get('cells', [])):
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'{name}: cell {i} contains execution results')
            if cell.get('metadata') or cell.get('attachments'):
                errors.append(f'{name}: cell {i} contains metadata or attachments')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print('Notebook hygiene passed; source contents still need human review.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
