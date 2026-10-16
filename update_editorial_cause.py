"""Update one editorial sentence from an already downloaded original; no refetch."""
import json,shutil
from pathlib import Path
from corpus import DATA,make_chunks
snapshot=DATA/'snapshots/before-lf-v1';snapshot.mkdir(parents=True,exist_ok=True)
for name in ['documents.json','chunks.json']:
 if not (snapshot/name).exists():shutil.copy2(DATA/name,snapshot/name)
docs=json.loads((DATA/'documents.json').read_text(encoding='utf-8'))
food=next(d for d in docs if d['id']=='food')
original=next(s for s in food['sections'] if 'cross-contamination' in s['section'])
assert 'spread germs' in original['text'] and 'cutting board' in original['text']
old='处理生肉与即食食品应使用不同的砧板。'
new='处理生肉与即食食品应使用不同的砧板，以免生肉上的病菌污染即食食品。'
food['summary']=food['summary'].replace(old,new)
food['sections'][0]['text']=food['sections'][0]['text'].replace(old,new)
food['editorial_revision']='2026-10-16; cause translated from original Separate foods to avoid cross-contamination section; not clinician reviewed'
(DATA/'documents.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding='utf-8')
chunks=[]
for doc in docs:chunks.extend(make_chunks(doc,doc['sections']))
(DATA/'chunks.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf-8')
print('editorial updated; original source bytes/hash unchanged; chunks',len(chunks))
