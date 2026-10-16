"""Finite JSON language for one bounded format-repair generation."""
import itertools,json

def selection_trie(tokenizer,count):
    if not 0<=count<=16:raise ValueError('Selection count must be 0..16')
    candidates=[json.dumps({'sentences':list(ids)},separators=(',',':'))
                for n in range(min(3,count)+1) for ids in itertools.combinations(range(1,count+1),n)]
    encoded=tokenizer(candidates,add_special_tokens=False)['input_ids']
    root={}
    for tokens in encoded:
        node=root
        for token in tokens+[tokenizer.eos_token_id]:node=node.setdefault(int(token),{})
    return root

def allowed_next(root,prefix,eos):
    node=root
    for token in prefix:
        if int(token) not in node:return [eos]
        node=node[int(token)]
    return list(node) or [eos]


def selection_gbnf(count):
    """Exact GBNF for the same finite language: at most 3 unique ids within 1..count.

    Enumerating the strings (instead of a generic array schema) keeps the retry
    guaranteed-unique and in-range, matching the token trie used by the
    Transformers backend. llama.cpp's GBNF parser only treats backslash and a few
    letters as escapes, so quoting the double quotes is all that is required.
    """
    if not 0<=count<=16:raise ValueError('Selection count must be 0..16')
    alternatives=[json.dumps({'sentences':list(ids)},separators=(',',':')).replace('"','\\"')
                  for n in range(min(3,count)+1) for ids in itertools.combinations(range(1,count+1),n)]
    return 'root ::= '+' | '.join('"'+literal+'"' for literal in alternatives)
