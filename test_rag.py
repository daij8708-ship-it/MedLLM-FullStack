import json
import re
import threading
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import rag_api
from corpus import extract_sections,make_chunks
from rag_core import RAGService,rewrite_query,resolve_citations,selected_evidence,evidence_sentences,missing_evidence_requirement
from query_plan import split_questions,personal_diagnosis,missing_detail,evidence_requirements,evidence_alignment,combine_results
from selection_constraints import selection_trie,allowed_next,selection_gbnf
from workspace import HOME, RAG


class WorkspaceTests(unittest.TestCase):
    """每个模块必须把工作目录映射到同一个根下的正确子目录。"""

    def test_every_module_agrees_on_the_rag_root(self):
        import corpus, evidence_index, run_span_eval
        from workspace import DATA, RUNS
        self.assertEqual(RAG, HOME / 'rag')
        self.assertEqual(DATA, HOME / 'data')
        self.assertEqual(RUNS, HOME / 'runs')
        for module in (corpus, evidence_index, run_span_eval, rag_api):
            self.assertEqual(module.DATA, RAG, module.__name__)

    def test_corpus_exposes_the_workspace_root(self):
        import corpus
        self.assertEqual(corpus.HOME, HOME)



class SelectionTests(unittest.TestCase):
    def test_alignment_object_and_same_sentence(self):
        self.assertIsNotNone(evidence_alignment('发电机的排气管应有多长？',[{'text':'发电机应离门窗20英尺。'}]))
        self.assertIsNone(evidence_alignment('发电机的排气管应有多长？',[{'text':'排气管长度为2米。'}]))
        self.assertIsNotNone(evidence_alignment('空气过滤器应该多久清洗一次？',[{'text':'空气净化器不能消除二手烟。成人每天清洁牙缝。'}]))
        self.assertIsNone(evidence_alignment('空气过滤器应该多久清洗一次？',[{'text':'空气净化器每月清洗一次。'}]))
    def test_json_constraint_language(self):
        class Tokenizer:
            eos_token_id=0
            def __call__(self,strings,**kwargs):return {'input_ids':[[ord(c) for c in s] for s in strings]}
        trie=selection_trie(Tokenizer(),2)
        self.assertEqual(allowed_next(trie,[],0),[ord('{')])
        prefix=[ord(c) for c in '{"sentences":[']
        self.assertEqual(set(allowed_next(trie,prefix,0)),{ord(']'),ord('1'),ord('2')})
        valid=[ord(c) for c in '{"sentences":[1,2]}']
        self.assertEqual(allowed_next(trie,valid,0),[0])
    def test_one_format_retry_can_recover(self):
        chunk={'id':'real','text':'资料记录了足够长度的相关内容。','metadata':{'kind':'editorial_summary'},'score':.9}
        class Retrieval:
            def search(self,*args,**kwargs):return [chunk]
            def expand_context(self,hits):return hits
            def encode(self,texts):return [[1.,0.] for _ in texts]
        class Generation:
            calls=0
            def generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls==2:assert kwargs['selection_count']==1
                return {'text':'invalid' if self.calls==1 else '{"sentences":[1]}','warnings':[],'cancelled':False,'tokens':2,'seconds':.1,'truncated':False,'citations':[]}
        service=RAGService();service.retriever=Retrieval();service.generator=Generation()
        r=service.ask('资料说明了什么内容？','rag')
        self.assertFalse(r['abstained']);self.assertEqual(service.generator.calls,2)
        self.assertEqual(len(r['selection_attempts']),2);self.assertTrue(r['format_retry'])
    def test_invalid_selection_abstains_without_excerpt_answer(self):
        chunk={'id':'real','text':'资料记录了足够长度的相关内容。','metadata':{'kind':'editorial_summary'},'score':.9}
        class Retrieval:
            def search(self,*args,**kwargs):return [chunk]
            def expand_context(self,hits):return hits
            def encode(self,texts):return [[1.,0.] for _ in texts]
        class Generation:
            def generate(self,*args,**kwargs):return {'text':'not JSON','warnings':[],'cancelled':False,'tokens':2,'seconds':.1,'truncated':False,'citations':[]}
        service=RAGService();service.retriever=Retrieval();service.generator=Generation()
        r=service.ask('资料说明了什么内容？','rag')
        self.assertTrue(r['abstained']);self.assertEqual(r['citations'],[])
        self.assertEqual(r['delivery_mode'],'evidence_selection_rejected')
    def test_requested_evidence_facets(self):
        self.assertIsNotNone(evidence_requirements('材料说多久换新？',[{'text':'每天清洁。'}]))
        self.assertIsNotNone(evidence_requirements('设备功率多大？',[{'text':'应在室外使用。'}]))
        self.assertIsNone(evidence_requirements('设备功率多大？',[{'text':'功率为500瓦。'}]))
        self.assertIsNotNone(evidence_requirements('材料使用几个月？',[{'text':'每天清洁两次。'}]))
        self.assertIsNone(evidence_requirements('操作应持续多长时间？',[{'text':'操作持续20秒。'}]))
    def test_query_planning_and_boundaries(self):
        self.assertEqual(split_questions('请分别说明刷牙次数，以及发电机应离窗户多远。'),['刷牙次数','发电机应离窗户多远'])
        self.assertEqual(len(split_questions('易腐食品常温下多久冷藏？超过32摄氏度时呢？')),1)
        self.assertTrue(personal_diagnosis('我口渴，是不是得糖尿病了？'))
        self.assertFalse(personal_diagnosis('糖尿病前期的资料介绍了什么？'))
        self.assertIsNotNone(missing_detail('牙刷充电需要多久？',[{'text':'每天刷牙两次。'}]))
    def test_merge_reference_remapping(self):
        def result(ident,text):return {'model':'rag','text':text+' [1]','citations':[{'id':ident,'label':1}],
            'retrieved':[],'abstained':False,'seconds':1,'tokens':2,'truncated':False,'cancelled':False}
        merged=combine_results('问题',['甲问题','乙问题'],[result('a','甲资料'),result('b','乙资料')])
        self.assertIn('乙资料 [2]',merged['text'])
        self.assertEqual([c['label'] for c in merged['citations']],[1,2])
    def test_missing_specific_evidence(self):
        chunks=[{'text':'每天使用含氟牙膏刷牙两次。'}]
        self.assertIsNotNone(missing_evidence_requirement('每次应该使用多少克牙膏？',chunks))
        self.assertIsNotNone(missing_evidence_requirement('明天温度是多少？',chunks))
        self.assertIsNone(missing_evidence_requirement('用多少克？',[{'text':'使用2克。'}]))
        self.assertIsNone(missing_evidence_requirement('每天刷牙几次？',chunks))
    def test_exact_selection_and_invalid_ids(self):
        chunks=[{'id':'c1','text':'第一条资料不能改。第二条资料也不能改。','metadata':{'kind':'editorial_summary'}}]
        options=evidence_sentences(chunks)
        text,cites=selected_evidence('{"sentences":[2]}',options,chunks)
        self.assertEqual(text,'第二条资料也不能改。 [1]')
        self.assertEqual(cites[0]['id'],'c1')
        for raw in ['{"sentences":[0]}','{"sentences":[99]}','{"sentences":[true]}','{"sentences":[1,1]}','自行编造回答']:
            self.assertIsNone(selected_evidence(raw,options,chunks))
        self.assertEqual(selected_evidence('{"sentences":[]}',options,chunks),('',[]))
        self.assertIsNotNone(selected_evidence('```json\n{"sentences":[1]}\n```',options,chunks))


class FakeService:
    generator=None;retriever=None;threshold=.45
    def ask(self,q,mode,history,cancel,stage):
        if q=='hold':cancel.wait(2)
        query,changed=rewrite_query(q,history)
        return {'model':mode,'text':'test [1]','tokens':4,'seconds':.01,'warnings':[],
                'retrieved':[],'citations':[],'rewritten':changed,'retrieval_query':query,'cancelled':cancel.is_set()}


class PureTests(unittest.TestCase):
    def test_parse_sections(self):
        sections=extract_sections('# 标题\n正文内容包含足够的文字，用于验证章节提取和元数据保存。'.encode(),'a.md')
        self.assertEqual(sections[0]['section'],'标题')
        doc={'id':'x','title':'test','url':'local:test','updated':'2026-01-01'}
        chunks=make_chunks(doc,sections)
        self.assertEqual(chunks[0]['metadata']['updated'],'2026-01-01')
        self.assertEqual(chunks[0]['metadata']['section'],'标题')

    def test_html_scripts_excluded(self):
        sections=extract_sections(b'<main><h2>Section</h2><p>Evidence text goes here.</p><script>alert(1)</script></main>','x.html')
        self.assertNotIn('alert',str(sections))

    def test_followup_rewrite(self):
        query,changed=rewrite_query('那一年要检查几次？',[{'question':'成人如何保护牙齿','answer':'刷牙'}])
        self.assertTrue(changed);self.assertIn('牙齿',query)
        self.assertEqual(rewrite_query('全新的完整问题是什么',[]),('全新的完整问题是什么',False))

    def test_citations_never_invented(self):
        docs=[{'id':'real','text':'proof','metadata':{}}]
        cites,invalid=resolve_citations('answer [1] [99]',docs)
        self.assertEqual(cites[0]['id'],'real');self.assertEqual(invalid,[99])


class API(unittest.TestCase):
    def setUp(self):
        self.old=rag_api.SERVICE;rag_api.SERVICE=FakeService();self.client=TestClient(rag_api.app)
    def tearDown(self):rag_api.SERVICE=self.old;self.client.close()
    def wait(self,ident):
        for _ in range(100):
            d=self.client.get('/api/jobs/'+ident).json()
            if d['state']!='running':return d
            time.sleep(.01)
        self.fail('timeout')
    def test_invalid(self):
        self.assertEqual(self.client.post('/api/jobs',json={'prompt':'a','mode':'invalid'}).status_code,400)
        self.assertEqual(self.client.post('/api/jobs',json={'prompt':''}).status_code,422)
    def test_context_and_reset(self):
        a=self.client.post('/api/jobs',json={'prompt':'成人如何保护牙齿','mode':'rag'}).json();self.wait(a['id'])
        b=self.client.post('/api/jobs',json={'prompt':'那一年几次','mode':'rag','session_id':a['session_id']}).json()
        d=self.wait(b['id']);self.assertTrue(d['results'][0]['rewritten']);self.assertIn('牙齿',d['results'][0]['retrieval_query'])
        self.assertEqual(self.client.delete('/api/sessions/'+a['session_id']).status_code,200)
        self.assertEqual(self.client.post('/api/jobs',json={'prompt':'next','session_id':a['session_id']}).status_code,404)
        self.assertFalse(rag_api.GPU.locked())
    def test_busy_cancel(self):
        a=self.client.post('/api/jobs',json={'prompt':'hold','mode':'base'}).json()
        self.assertEqual(self.client.post('/api/jobs',json={'prompt':'next'}).status_code,409)
        self.client.post('/api/cancel/'+a['id']);self.assertEqual(self.wait(a['id'])['state'],'cancelled')
    def test_cross_site(self):
        self.assertEqual(self.client.post('/api/jobs',json={'prompt':'x'},headers={'Origin':'https://other.example'}).status_code,403)
    def test_upload_permission(self):
        result=self.client.post('/api/documents',data={'title':'test','permission':'false'},files={'file':('x.txt',b'test')})
        self.assertEqual(result.status_code,400)


class LlamaBackendTests(unittest.TestCase):
    """Runs the llama.cpp backend against a stub llama-server (no GPU, no network)."""

    @classmethod
    def setUpClass(cls):
        import http.server, json, socketserver, os
        cls.requests = []
        cls.reply = ['{"sentences":[1,2]}']
        cls.delay = 0.0
        requests = cls.requests

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *a): pass
            def _send(self, body, ctype='application/json'):
                raw = body.encode('utf-8')
                self.send_response(200); self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def do_GET(self):
                if self.path == '/health': self._send('{"status":"ok"}')
                else: self.send_response(404); self.end_headers()
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                requests.append(json.loads(raw))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                try:
                    for piece in cls.reply:
                        if cls.delay: time.sleep(cls.delay)
                        chunk = {'choices': [{'delta': {'content': piece}}]}
                        self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                        self.wfile.flush()
                    self.wfile.write(('data: ' + json.dumps({'choices': [], 'timings': {'predicted_n': 9, 'prompt_n': 65}}) + '\n\n').encode())
                    self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass  # client aborted the stream on purpose (cancel test)

        cls.server = socketserver.TCPServer(('127.0.0.1', 0), Handler)
        cls.url = f'http://127.0.0.1:{cls.server.server_address[1]}'
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        os.environ['RAG_LLAMA_BASE_URL'] = cls.url
        os.environ['RAG_LLAMA_SFT_URL'] = cls.url
        os.environ.pop('RAG_LLAMA_TOKENIZER_DIR', None)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()

    def setUp(self):
        import llama_backend
        self.backend = llama_backend
        self.cls = type(self)
        self.cls.requests.clear(); self.cls.reply = ['{"sentences":[1,2]}']; self.cls.delay = 0.0

    def test_gbnf_enumerates_exact_language(self):
        for count in (0, 1, 2, 3, 5):
            grammar = selection_gbnf(count)
            literals = re.findall(r'"((?:[^"\\]|\\.)*)"', grammar)
            expected = 1 + count + count * (count - 1) // 2 + count * (count - 1) * (count - 2) // 6
            self.assertEqual(len(literals), expected)
            for literal in literals:
                ids = json.loads(literal.replace('\\"', '"'))['sentences']
                self.assertLessEqual(len(ids), 3)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(1 <= n <= count for n in ids))
        with self.assertRaises(ValueError): selection_gbnf(17)

    def test_stream_request_and_grammar_payload(self):
        result = self.backend.LlamaServerGenerator().generate('问题','sft',max_tokens=32,selection_count=2)
        self.assertEqual(result['text'], '{"sentences":[1,2]}')
        self.assertEqual(result['tokens'], 9)
        self.assertEqual(result['backend'], 'llamacpp')
        self.assertFalse(result['cancelled'])
        sent = self.cls.requests[-1]
        self.assertIn('grammar', sent)
        self.assertNotIn('prompt', sent)
        self.assertEqual(sent['messages'][-1]['role'], 'user')
        self.assertEqual(sent['temperature'], 0)

    def test_think_block_is_stripped(self):
        self.cls.reply = ['<think>', '推理中', '</think>', '\n{"sentences":[1]}']
        result = self.backend.LlamaServerGenerator().generate('问题','base',max_tokens=32)
        self.assertEqual(result['text'], '{"sentences":[1]}')

    def test_cancel_aborts_stream(self):
        self.cls.reply = ['x'] * 50
        self.cls.delay = 0.02
        cancel = threading.Event()
        threading.Timer(0.15, cancel.set).start()
        result = self.backend.LlamaServerGenerator().generate('问题','base',max_tokens=64,cancel=cancel)
        self.assertTrue(result['cancelled'])

    def test_dead_server_raises(self):
        import os
        old = os.environ['RAG_LLAMA_SFT_URL']
        os.environ['RAG_LLAMA_SFT_URL'] = 'http://127.0.0.1:9'
        try:
            with self.assertRaises(self.backend.LlamaServerError):
                self.backend.LlamaServerGenerator()
        finally:
            os.environ['RAG_LLAMA_SFT_URL'] = old


class SpanMetricTests(unittest.TestCase):
    def setUp(self):
        import span_metrics
        self.m = span_metrics

    def test_wilson_interval_shape(self):
        self.assertEqual(self.m.wilson(0, 0), (None, None))
        lo, hi = self.m.wilson(4, 8)
        self.assertTrue(0 <= lo < 0.5 < hi <= 1)
        self.assertEqual(self.m.wilson(10, 10)[1], 1.0)

    def test_perfect_and_partial_selection(self):
        perfect = self.m.selection_metrics(['a'], ['a'], ['a'])
        self.assertEqual((perfect['precision'], perfect['recall'], perfect['f1']), (1.0, 1.0, 1.0))
        self.assertTrue(perfect['exact'])
        partial = self.m.selection_metrics(['a', 'b'], ['a', 'b'], ['a'])
        self.assertEqual(partial['recall'], 0.5)
        self.assertEqual(partial['padding'], 0)
        self.assertEqual(partial['off_topic'], 0)
        self.assertFalse(partial['exact'])
        self.assertTrue(partial['any_hit'])

    def test_padding_and_off_topic_are_distinguished(self):
        # 'ctx' is the same document's adjacent sentence, 'alien' is another document's.
        result = self.m.selection_metrics(['a'], ['a'], ['a', 'ctx', 'alien'],
                                         doc_scope=['a', 'ctx'])
        self.assertEqual(result['padding'], 2)
        self.assertEqual(result['off_topic'], 1)
        self.assertEqual(result['precision'], round(1 / 3, 4))
        self.assertEqual(result['lenient_precision'], round(2 / 3, 4))
        self.assertFalse(result['exact'])

    def test_confusion_separates_both_directions(self):
        rows = [{'expected': 'refuse', 'abstained': True},   # correct refusal
                {'expected': 'answer', 'abstained': True},   # over refusal
                {'expected': 'refuse', 'abstained': False},  # under refusal
                {'expected': 'answer', 'abstained': False}]
        c = self.m.confusion(rows)
        self.assertEqual((c['correct_refusal'], c['over_refusal'], c['under_refusal'],
                          c['answered_when_answerable']), (1, 1, 1, 1))
        self.assertEqual(c['over_refusal_rate'], 0.5)
        self.assertEqual(c['under_refusal_rate'], 0.5)

    def test_errors_are_excluded_not_silently_zeroed(self):
        cases = [{'expected': 'answer', 'abstained': False, 'core': ['a'], 'acceptable': ['a'],
                  'selected': ['a'], 'error': False},
                 {'expected': 'answer', 'abstained': False, 'core': ['a'], 'acceptable': ['a'],
                  'selected': [], 'error': True}]
        report = self.m.aggregate(cases)
        self.assertEqual(report['errors'], 1)
        self.assertEqual(report['answerable'], 1)
        self.assertEqual(report['precision'], 1.0)
        self.assertEqual(report['coverage'], 1.0)

    def test_reachability_distinguishes_retrieval_from_selection(self):
        self.assertTrue(self.m.candidate_reachability(['a'], ['a', 'b'])['reachable'])
        self.assertFalse(self.m.candidate_reachability(['a'], ['b', 'c'])['reachable'])
        self.assertEqual(self.m.candidate_reachability([], [])['reachable'], True)


class GoldIntegrityTests(unittest.TestCase):
    """The gold artefact must stay a subset of the frozen corpus."""

    GOLD = RAG / 'gold' / 'evidence-gold-v1.json'
    INDEX = RAG / 'gold' / 'sentence-index.json'

    def setUp(self):
        if not self.GOLD.exists() or not self.INDEX.exists():
            self.skipTest('gold artefacts not built in this environment')
        self.gold = json.loads(self.GOLD.read_text(encoding='utf-8'))
        self.index = json.loads(self.INDEX.read_text(encoding='utf-8'))

    def test_corpus_hash_matches_index(self):
        self.assertEqual(self.gold['corpus_sha256'], self.index['corpus_sha256'])

    def test_every_gold_key_exists_in_corpus(self):
        known = {e['key'] for e in self.index['index']}
        for case in self.gold['cases']:
            for key in case['core'] + case['acceptable']:
                self.assertIn(key, known, f"{case['suite']}/{case['id']}")

    def test_core_is_subset_of_acceptable(self):
        for case in self.gold['cases']:
            self.assertTrue(set(case['core']) <= set(case['acceptable']), f"{case['suite']}/{case['id']}")

    def test_answerable_cases_need_core_and_refusals_need_none(self):
        for case in self.gold['cases']:
            if case['expected'] == 'answer':
                self.assertTrue(case['core'], f"{case['suite']}/{case['id']} has no core")
            else:
                self.assertEqual(case['core'], [])

    def test_keys_are_stable_under_whitespace(self):
        import evidence_index
        self.assertEqual(evidence_index.sentence_key('c', ' 甲。 \n 乙。 '),
                         evidence_index.sentence_key('c', '甲。 乙。'))


if __name__=='__main__':unittest.main()
