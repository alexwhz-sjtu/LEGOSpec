import json,tempfile
from pathlib import Path
from jsonl_io import read_jsonl

with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'unicode.jsonl'
    expected=[{'id':1,'text':'head\u2028middle\u2029tail\u0085last\ninside'},
              {'id':2,'text':'ordinary'}]
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in expected),encoding='utf-8')
    assert list(read_jsonl(path))==expected
print('Unicode U+2028/U+2029/NEL preserved inside JSON strings; LF-only records PASS')
