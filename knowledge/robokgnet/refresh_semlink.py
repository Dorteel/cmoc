"""Explicit online refresh, reusing CMOC's SemLink loader; normal builds are offline."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from nltk.corpus import verbnet
from tools.semantic_bridge import _load_semlink, SEMLINK_URL


def main():
    mappings = _load_semlink()
    records, missing = [], []
    for class_id in ('bring-11.3', 'bring-11.3-1', 'search-35.2'):
        for member in verbnet.vnclass(class_id).findall('MEMBERS/MEMBER'):
            lemma = member.get('name')
            frames = mappings.get((verbnet.shortid(class_id), lemma), [])
            if not frames:
                missing.append(f'{class_id}/{lemma}')
            for frame in frames:
                records.append(dict(verbnet=class_id, member=lemma, framenet=frame))
    canonical = json.dumps(sorted((c, m, f) for (c, m), f in mappings.items()))
    data = dict(source=SEMLINK_URL, retrieved=datetime.now(timezone.utc).isoformat(),
                parsed_table_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                alignments=records, unmapped_members=missing,
                note='Exact class/member mappings only; source contains no role-level alignment.')
    path = Path(__file__).parent / 'data/semlink_subset.json'
    path.write_text(json.dumps(data, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
