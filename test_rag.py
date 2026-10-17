import json
import re
import threading
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import rag_api
from corpus import extract_sections,make_chunks
from errors import ContextOverflowError
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

    GOLD = RAG / 'gold' / 'evidence-gold-v3.json'
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

    def test_doc_scope_covers_more_than_the_editorial_summary(self):
        """v2 的宽松分母必须是答案文档全部句子，否则 lenient_precision 又是恒等式。"""
        summary = {e['key'] for e in self.index['index'] if e['in_editorial_summary']}
        answerable = [c for c in self.gold['cases'] if c['expected'] == 'answer']
        self.assertTrue(answerable)
        wide = [c for c in answerable if set(c['doc_scope']) - summary]
        self.assertEqual(len(wide), len(answerable), '每道可答题的 doc_scope 都应包含非导读句')
        for case in answerable:
            self.assertTrue(set(case['acceptable']) <= set(case['doc_scope']), f"{case['suite']}/{case['id']}")

    def test_expansion_suite_is_present_and_sized_as_documented(self):
        """扩充集必须在产物里、题数与 builder 自报的数字一致。"""
        self.assertEqual(self.gold['name'], 'evidence-gold-v3')
        sub = [c for c in self.gold['cases'] if c['suite'] == 'expansion']
        self.assertEqual(len(sub), 56)
        self.assertEqual(sum(c['expected'] == 'answer' for c in sub), 38)
        self.assertEqual(sum(c['expected'] == 'refuse' for c in sub), 18)

    def test_questions_are_unique_across_suites(self):
        questions = [c['question'] for c in self.gold['cases']]
        duplicates = {q for q in questions if questions.count(q) > 1}
        self.assertEqual(duplicates, set())

    def test_some_answer_cases_need_several_sentences(self):
        """core 全是一句的话召回分辨率只能取 0/0.5/1，扩充集必须写出多句事实单元。"""
        multi = [c for c in self.gold['cases'] if c['expected'] == 'answer' and len(c['core']) > 1]
        self.assertGreaterEqual(len(multi), 8)
        self.assertEqual(self.gold['counts']['answer_cases_needing_multiple_sentences'], len(multi))

    def test_expansion_source_ships_with_the_repo_and_marks_refusals(self):
        """扩充集的问题清单要随仓库分发，且每题都显式写 refuse，不留静默缺省。"""
        source = Path(__file__).resolve().parent / 'evidence_expansion_cases.json'
        self.assertTrue(source.exists())
        cases = json.loads(source.read_text(encoding='utf-8'))
        self.assertEqual(len(cases), 56)
        for case in cases:
            self.assertIn('refuse', case, case['id'])
            self.assertIsInstance(case['refuse'], bool, case['id'])
            self.assertTrue(case['q'].strip(), case['id'])

    def test_unlabelled_case_without_refusal_flag_is_rejected(self):
        """未标注又不标 refuse 的案例以前会被静默跳过，现在必须报错。"""
        import gold_span
        with self.assertRaises(SystemExit):
            gold_span.case_kind('expansion', 'zz', {'id': 'zz', 'q': '测试'})
        self.assertEqual(gold_span.case_kind('expansion', 'zz', {'id': 'zz', 'refuse': True}), 'refuse')
        self.assertEqual(gold_span.case_kind('expansion', 'e01', {'id': 'e01'}), 'answer')


class CandidateScopeTests(unittest.TestCase):
    """The selector's candidate universe is switchable and must stay auditable."""

    CHUNKS = [{'text': '成人应每天使用含氟牙膏认真刷牙两次，并使用牙线清洁牙缝以去除牙菌斑。即使戴假牙，也应至少每年看一次牙医。',
               'metadata': {'kind': 'editorial_summary'}},
              {'text': 'Wash your hands before, during, and after preparing food. Keep raw meat away from other foods.',
               'metadata': {'kind': 'original'}}]

    def test_summary_only_skips_later_non_summary_chunks(self):
        from rag_core import evidence_sentences
        options = evidence_sentences(self.CHUNKS, 'summary_only')
        self.assertEqual({o['chunk_index'] for o in options}, {0})

    def test_retrieved_sentences_lets_every_chunk_contribute(self):
        from rag_core import evidence_sentences
        options = evidence_sentences(self.CHUNKS, 'retrieved_sentences')
        self.assertEqual({o['chunk_index'] for o in options}, {0, 1})
        self.assertEqual([o['option'] for o in options], list(range(1, len(options) + 1)))

    def test_unknown_scope_is_rejected_not_silently_defaulted(self):
        from rag_core import evidence_sentences
        with self.assertRaises(ValueError):
            evidence_sentences(self.CHUNKS, 'everything')

    def test_english_sentences_split_on_period_but_not_on_decimals(self):
        import evidence_index
        parts = evidence_index.split_sentences(
            'Refrigerate at 40°F (about 4°C). Bacteria grow fast above that. Use 2.5 hours as a limit.')
        self.assertEqual(len(parts), 3)
        self.assertIn('Use 2.5 hours as a limit.', parts)

    def test_service_rejects_a_bad_candidate_scope(self):
        import os, rag_core
        os.environ['RAG_CANDIDATE_SCOPE'] = 'everything'
        try:
            with self.assertRaises(ValueError):
                rag_core.RAGService()
        finally:
            os.environ.pop('RAG_CANDIDATE_SCOPE', None)

    def test_auto_keeps_narrow_scope_for_a_document_with_a_summary(self):
        from rag_core import resolve_candidate_scope
        chunks = [{'text': '导读。', 'metadata': {'doc_id': 'cdc', 'kind': 'editorial_summary'}},
                  {'text': '正文。', 'metadata': {'doc_id': 'cdc', 'kind': 'original'}}]
        self.assertEqual(resolve_candidate_scope(chunks, 'auto'), 'summary_only')

    def test_auto_widens_scope_for_an_upload_without_a_summary(self):
        from rag_core import resolve_candidate_scope
        chunks = [{'text': '正文。', 'metadata': {'doc_id': 'upload_1', 'kind': 'original'}}]
        self.assertEqual(resolve_candidate_scope(chunks, 'auto'), 'retrieved_sentences')

    def test_auto_is_decided_by_the_top_document_only(self):
        from rag_core import resolve_candidate_scope
        chunks = [{'text': '上传正文。', 'metadata': {'doc_id': 'upload_1', 'kind': 'original'}},
                  {'text': 'CDC 导读。', 'metadata': {'doc_id': 'food', 'kind': 'editorial_summary'}}]
        self.assertEqual(resolve_candidate_scope(chunks, 'auto'), 'retrieved_sentences')

    def test_explicit_scope_is_not_rewritten(self):
        from rag_core import resolve_candidate_scope
        chunks = [{'text': '正文。', 'metadata': {'doc_id': 'upload_1', 'kind': 'original'}}]
        self.assertEqual(resolve_candidate_scope(chunks, 'summary_only'), 'summary_only')
        self.assertEqual(resolve_candidate_scope(chunks, 'retrieved_sentences'), 'retrieved_sentences')

    def test_auto_is_not_a_concrete_selector_scope(self):
        from rag_core import evidence_sentences
        with self.assertRaises(ValueError):
            evidence_sentences(self.CHUNKS, 'auto')


class ReflowTests(unittest.TestCase):
    """PDF/OCR 的硬换行必须在切句前还原。"""

    def test_dehyphenates_a_wrapped_word(self):
        import corpus
        text = corpus.reflow_text('The 2025 adult immuniza-\ntion schedule can be found.')
        self.assertIn('immunization schedule', text)
        self.assertNotIn('immuniza-\n', text)

    def test_joins_wrapped_latin_lines_with_one_space(self):
        import corpus
        text = corpus.reflow_text('The schedule supports health care provid-\ners, as well as public health.')
        self.assertEqual(text, 'The schedule supports health care providers, as well as public health.')

    def test_joins_wrapped_chinese_without_a_space(self):
        import corpus
        text = corpus.reflow_text('高血壓是指以體循環動脈血壓\n持續升高為主要表現。')
        self.assertEqual(text, '高血壓是指以體循環動脈血壓持續升高為主要表現。')

    def test_keeps_page_numbers_table_rows_and_chart_labels_isolated(self):
        import corpus
        text = corpus.reflow_text('Morbidity and Mortality Weekly Report\n30\n'
                                  'U.S. Department of Health  |  CDC\n'
                                  'SSA 67.17%\n$212.6\nThis OIG report describes the period.')
        lines = text.splitlines()
        self.assertIn('30', lines)
        self.assertIn('SSA 67.17%', lines)
        self.assertIn('$212.6', lines)
        self.assertIn('This OIG report describes the period.', lines)

    def test_keeps_blank_line_paragraph_breaks(self):
        import corpus
        self.assertEqual(corpus.reflow_text('第一段。\n\n第二段。'), '第一段。\n\n第二段。')

    def test_txt_input_is_not_reflowed(self):
        import corpus
        sections = corpus.extract_sections(b'line one\nline two', 'sample.txt')
        self.assertEqual(sections[0]['text'], 'line one\nline two')


class MinScoreTests(unittest.TestCase):
    """检索阈值是拒答的第一道闸：缺省值有依据、写错必须拒绝启动、边界行为可测。"""

    @staticmethod
    def _service(score):
        chunk = {'id': 'c1', 'text': '成人应每天使用含氟牙膏认真刷牙两次。',
                 'metadata': {'kind': 'editorial_summary'}, 'score': score}

        class Retrieval:
            def search(self, *args, **kwargs): return [chunk]
            def expand_context(self, hits): return hits
            def encode(self, texts): return [[1., 0.] for _ in texts]

        class Generation:
            def generate(self, *args, **kwargs):
                return {'text': '{"sentences":[1]}', 'warnings': [], 'cancelled': False,
                        'tokens': 2, 'seconds': .1, 'truncated': False, 'citations': []}

        service = RAGService()
        service.retriever = Retrieval()
        service.generator = Generation()
        return service

    def test_default_is_the_measured_operating_point(self):
        import os, rag_core
        saved = os.environ.pop('RAG_MIN_SCORE', None)
        try:
            self.assertEqual(rag_core.RAGService().threshold, rag_core.DEFAULT_MIN_SCORE)
            self.assertEqual(rag_core.DEFAULT_MIN_SCORE, 0.48)
        finally:
            if saved is not None: os.environ['RAG_MIN_SCORE'] = saved

    def test_explicit_value_is_honoured(self):
        import os, rag_core
        os.environ['RAG_MIN_SCORE'] = '0.52'
        try:
            self.assertEqual(rag_core.RAGService().threshold, 0.52)
        finally:
            os.environ.pop('RAG_MIN_SCORE', None)

    def test_bad_values_are_rejected_rather_than_defaulted(self):
        import os, rag_core
        for bad in ('abc', '0.05', '0.99', ''):
            os.environ['RAG_MIN_SCORE'] = bad
            try:
                if bad == '':
                    self.assertEqual(rag_core.RAGService().threshold, rag_core.DEFAULT_MIN_SCORE)
                else:
                    with self.assertRaises(ValueError):
                        rag_core.RAGService()
            finally:
                os.environ.pop('RAG_MIN_SCORE', None)

    def test_gate_refuses_just_below_and_answers_just_above(self):
        below = self._service(0.479).ask('成人每天刷牙几次？', 'rag')
        self.assertTrue(below['abstained'])
        self.assertEqual(below['abstention_reason'], 'low_retrieval_score')
        above = self._service(0.481).ask('成人每天刷牙几次？', 'rag')
        self.assertFalse(above['abstained'])
        self.assertTrue(above['text'])


class RerankerTests(unittest.TestCase):
    """重排序决定哪些候选进模型视野，所以开关必须显式、可测、缺权重时明确报错。"""

    @staticmethod
    def _options():
        return [{'option': 1, 'chunk_index': 0, 'text': '甲句', 'sentence_score': 0.9},
                {'option': 2, 'chunk_index': 0, 'text': '乙句', 'sentence_score': 0.5},
                {'option': 3, 'chunk_index': 1, 'text': '丙句', 'sentence_score': 0.1}]

    def test_none_keeps_dense_order_and_truncates(self):
        from reranker import SentenceReranker, rank_options
        kept = rank_options(self._options(), SentenceReranker('none'), 'q', 2)
        self.assertEqual([o['text'] for o in kept], ['甲句', '乙句'])
        self.assertNotIn('rerank_score', kept[0])

    def test_cross_encoder_can_rescue_a_low_dense_candidate(self):
        from reranker import rank_options

        class Fake:
            active = True

            def scores(self, query, texts):
                return [0.1, 0.2, 3.0]

        kept = rank_options(self._options(), Fake(), 'q', 2)
        self.assertEqual([o['text'] for o in kept], ['丙句', '乙句'])
        # 截断在打分之后，否则低稠密分的候选永远进不了模型视野。
        self.assertEqual(kept[0]['rerank_score'], 3.0)

    def test_unknown_reranker_is_rejected_not_silently_defaulted(self):
        import os, reranker
        os.environ['RAG_RERANKER'] = 'cross_encoder'
        try:
            with self.assertRaises(ValueError):
                reranker.reranker_from_env()
        finally:
            os.environ.pop('RAG_RERANKER', None)

    def test_service_rejects_a_bad_reranker(self):
        import os, rag_core
        os.environ['RAG_RERANKER'] = 'cross_encoder'
        try:
            with self.assertRaises(ValueError):
                rag_core.RAGService()
        finally:
            os.environ.pop('RAG_RERANKER', None)

    def test_missing_weights_fail_only_when_the_reranker_is_used(self):
        from reranker import SentenceReranker
        service = SentenceReranker('bge_reranker_v2_m3')
        self.assertTrue(service.active)
        self.assertIsNone(service.model)
        if service.path.exists():
            self.skipTest('重排序权重已存在，这里只验证缺权重时的报错')
        with self.assertRaises(RuntimeError):
            service.scores('q', ['句子'])


class ContextTrimTests(unittest.TestCase):
    """提示超限时按相关度裁剪候选并声明，而不是把错误直接抛给用户。"""

    @staticmethod
    def _options(count):
        return [{'option': i, 'chunk_index': 0, 'text': f'这是第{i}句候选，长度足够进入选句提示。',
                 'sentence_score': 1 - i / 10} for i in range(1, count + 1)]

    def _fake(self, fail_times):
        calls = []

        class FakeGenerator:
            def generate(self, task, role, cancel=None, max_tokens=0, selection_count=None):
                calls.append(task)
                if len(calls) <= fail_times:
                    raise ContextOverflowError('上下文超过 4096 token（提示 5000）')
                return {'text': '{"sentences":[1]}', 'tokens': 5, 'seconds': 0.1,
                        'warnings': [], 'cancelled': False}

        return FakeGenerator(), calls

    def test_trim_drops_the_lowest_scored_candidates_and_reports_the_count(self):
        import rag_core
        options = self._options(6)
        generator, calls = self._fake(fail_times=1)
        result, dropped = rag_core.generate_with_context_trim(generator, options, '问题', 'base', 64)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(options), 5)
        self.assertEqual([o['option'] for o in options], [1, 2, 3, 4, 5])
        self.assertIn('第5句', calls[1], '裁剪后重试的提示里应保留更相关的候选')
        self.assertNotIn('第6句', calls[1], '被丢弃的候选不能还留在提示里')
        self.assertEqual(result['text'], '{"sentences":[1]}')

    def test_trimming_stops_at_one_candidate_and_then_raises(self):
        import rag_core
        options = self._options(3)
        generator, calls = self._fake(fail_times=99)
        with self.assertRaises(ContextOverflowError):
            rag_core.generate_with_context_trim(generator, options, '问题', 'base', 64)
        self.assertEqual(len(options), 1, '只剩一句候选时必须停手，交给上层报错')
        self.assertEqual(len(calls), 3)


class PdfOcrTests(unittest.TestCase):
    """扫描件走 OCR 必须是显式开关；缺依赖时明确报错，不静默降级。"""

    @staticmethod
    def _textless_pdf():
        import io
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        buffer = io.BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    def test_default_mode_is_off(self):
        import corpus
        self.assertEqual(corpus.pdf_ocr_mode(), 'off')

    def test_unknown_mode_is_rejected_not_silently_ignored(self):
        import os, corpus
        os.environ['RAG_PDF_OCR'] = 'tesseract'
        try:
            with self.assertRaises(ValueError):
                corpus.pdf_ocr_mode()
        finally:
            os.environ.pop('RAG_PDF_OCR', None)

    def test_textless_pdf_is_rejected_with_the_switch_name(self):
        import corpus
        with self.assertRaises(ValueError) as ctx:
            corpus.extract_sections(self._textless_pdf(), 'scan.pdf')
        self.assertIn('RAG_PDF_OCR', str(ctx.exception))

    def test_ocr_enabled_without_dependencies_says_what_to_install(self):
        import os, corpus
        os.environ['RAG_PDF_OCR'] = 'rapidocr'
        try:
            try:
                import rapidocr_onnxruntime  # noqa: F401
                self.skipTest('本机已安装 rapidocr，缺依赖分支无法在这里复现')
            except ImportError:
                pass
            with self.assertRaises(ValueError) as ctx:
                corpus.extract_sections(self._textless_pdf(), 'scan.pdf')
            self.assertIn('requirements-ocr.txt', str(ctx.exception))
        finally:
            os.environ.pop('RAG_PDF_OCR', None)

    def test_only_pages_without_a_text_layer_go_to_ocr(self):
        """混合文档里，有文本层的页不能被 OCR 覆盖掉。"""
        import corpus
        pages = [{'page': 1, 'text': '这一页有文本层'}, {'page': 2, 'text': '  '},
                 {'page': 3, 'text': '第三页也有文本'}, {'page': 4, 'text': ''}]
        self.assertEqual(corpus.pages_needing_ocr(pages), [2, 4])




if __name__=='__main__':unittest.main()
