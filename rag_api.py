"""FastAPI local RAG service, reusing a single 4-bit base model."""
import asyncio
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from corpus import DATA,HOME
from rag_core import RAGService

ROOT=Path(__file__).resolve().parent
app=FastAPI(title='知微 · 证据问答服务',version='1.0',description='Local BGE-M3 + ChromaDB + PEFT research service. Not clinical advice.')
SERVICE=RAGService();GPU=threading.Lock();STATE=threading.Lock()
JOBS=OrderedDict();SESSIONS=OrderedDict()
MODES={'base','sft','rag','sft_rag','compare'}


@app.middleware('http')
async def local_only(request:Request,call_next):
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD') and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc!=request.headers.get('host'):
            return JSONResponse({'error':'不允许跨站写入。'},403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Cache-Control']='no-store'
    return response


class Ask(BaseModel):
    prompt:str=Field(min_length=1,max_length=1000)
    mode:str='rag'
    session_id:str|None=None


class Search(BaseModel):
    query:str=Field(min_length=1,max_length=1000)
    k:int=Field(default=3,ge=1,le=8)


def public(job):return {k:v for k,v in job.items() if not k.startswith('_')}


def run(job):
    try:
        with STATE:
            session=SESSIONS[job['session_id']]
            history=list(session['turns'])
        modes=['base','sft_rag'] if job['mode']=='compare' else [job['mode']]
        def stage(text):
            with STATE:job['stage']=text
        for mode in modes:
            if job['_cancel'].is_set():break
            result=SERVICE.ask(job['prompt'],mode,history,job['_cancel'],stage)
            with STATE:job['results'].append(result)
        with STATE:
            stopped=job['_cancel'].is_set()
            job['state']='cancelled' if stopped else 'done'
            job['stage']='已停止' if stopped else '回答完成，请核对来源与事实'
            if not stopped and job['results']:
                last=job['results'][-1]
                session['turns'].append({'question':job['prompt'],'answer':last['text']})
                session['turns']=session['turns'][-4:]
                session['updated']=time.monotonic()
    except Exception as exc:
        with STATE:job.update(state='error',stage='运行失败',error=str(exc))
    finally:GPU.release()


@app.get('/api/status')
def status():
    documents=json.loads((DATA/'documents.json').read_text(encoding='utf-8')) if (DATA/'documents.json').exists() else []
    uploads=list((DATA/'uploads').glob('*.json')) if (DATA/'uploads').exists() else []
    payload={'service':'fastapi-rag','ready':SERVICE.generator is not None,'busy':GPU.locked(),
        'embedding_model':'BAAI/bge-m3','embedding_device':'cpu','embedding_ready':SERVICE.retriever is not None,
        'documents':len(documents)+len(uploads),'chunks':SERVICE.retriever.collection.count() if SERVICE.retriever else None,
        'retrieval_threshold':SERVICE.threshold,'adapter':Path(os.environ.get('MEDLLM_ADAPTER',str(HOME/'runs/medical-v4'))).name,
        'answer_style':getattr(SERVICE,'answer_style','grounded_selection'),
        'generator_backend':SERVICE.backend}
    if SERVICE.backend=='llamacpp':
        from llama_backend import probe
        servers={}
        for role,key in (('base','RAG_LLAMA_BASE_URL'),('sft','RAG_LLAMA_SFT_URL')):
            url=os.environ.get(key,{'base':'http://127.0.0.1:8081','sft':'http://127.0.0.1:8080'}[role])
            ok,detail=probe(url);servers[role]={'url':url,'ready':ok,'detail':detail}
        served=Path(os.environ.get('RAG_LLAMA_SFT_MODEL',str(HOME/'gguf/evidence-lf-v2-Q4_K_M.gguf'))).name
        payload.update(ready=all(s['ready'] for s in servers.values()),llama_servers=servers,model_served=served)
    return payload


@app.post('/api/jobs',status_code=202)
def ask(body:Ask):
    if not body.prompt.strip():raise HTTPException(400,'问题不能为空。')
    if body.mode not in MODES:raise HTTPException(400,'未知模式。')
    if not GPU.acquire(blocking=False):raise HTTPException(409,'GPU或索引正在处理另一个任务，请稍后重试。')
    try:
        with STATE:
            now=time.monotonic()
            # Expire only inactive sessions; never evict the active job's context.
            for sid in list(SESSIONS):
                if now-SESSIONS[sid]['updated']>1800:SESSIONS.pop(sid,None)
            sid=body.session_id or uuid.uuid4().hex
            if body.session_id and sid not in SESSIONS:raise HTTPException(404,'会话已过期，请新建对话。')
            if sid not in SESSIONS:
                if len(SESSIONS)>=100:SESSIONS.popitem(last=False)
                SESSIONS[sid]={'turns':[],'updated':now}
            job={'id':uuid.uuid4().hex,'session_id':sid,'prompt':body.prompt.strip(),'mode':body.mode,
                'state':'running','stage':'准备运行…','results':[],'_cancel':threading.Event()}
            JOBS[job['id']]=job
            while len(JOBS)>30:JOBS.popitem(last=False)
        threading.Thread(target=run,args=(job,),daemon=True).start()
        return {'id':job['id'],'session_id':sid}
    except Exception:
        GPU.release();raise


@app.get('/api/jobs/{ident}')
def job(ident:str):
    with STATE:
        if ident not in JOBS:raise HTTPException(404,'任务不存在或已过期。')
        return public(JOBS[ident])


@app.post('/api/cancel/{ident}')
def cancel(ident:str):
    with STATE:
        if ident not in JOBS:raise HTTPException(404,'任务不存在。')
        if JOBS[ident]['state']=='running':
            JOBS[ident]['_cancel'].set();JOBS[ident]['stage']='正在停止；加载阶段需要等待完成…'
    return {'ok':True}


@app.delete('/api/sessions/{sid}')
def clear_session(sid:str):
    if GPU.locked():raise HTTPException(409,'请先等待或停止当前任务。')
    with STATE:SESSIONS.pop(sid,None)
    return {'ok':True}


@app.post('/api/search')
def search(body:Search):
    if not GPU.acquire(blocking=False):raise HTTPException(409,'服务正在运行，请稍后检索。')
    try:
        hits=SERVICE.retrieval_ready().search(body.query,body.k)
        return {'query':body.query,'hits':hits,'threshold':SERVICE.threshold}
    finally:GPU.release()


@app.get('/api/documents')
def documents():
    docs=json.loads((DATA/'documents.json').read_text(encoding='utf-8'))
    for p in (DATA/'uploads').glob('*.json') if (DATA/'uploads').exists() else []:
        docs.append(json.loads(p.read_text(encoding='utf-8')))
    return [{'id':d['id'],'title':d['title'],'source':d['url'],'updated':d['updated'],
             'published':d.get('published','unknown'),'last_published':d.get('last_published','unknown'),
             'license':d['license'],'fetched':d.get('fetched','unknown'),
             'sections':len(d.get('sections',[])),'chunks':d.get('chunk_count',len(d.get('chunks',[]))),
             'raw_sha256':d.get('raw_sha256','')} for d in docs]


@app.get('/api/chunks/{ident}')
def chunk(ident:str):
    # Access already-local vectors only; do not fetch arbitrary remote URLs.
    retriever=SERVICE.retrieval_ready()
    result=retriever.collection.get(ids=[ident],include=['documents','metadatas'])
    if not result['ids']:raise HTTPException(404,'段落不存在。')
    return {'id':ident,'text':result['documents'][0],'metadata':result['metadatas'][0]}


@app.post('/api/documents',status_code=201)
async def upload(file:UploadFile=File(...),title:str=Form(...),source:str=Form(''),updated:str=Form('unknown'),permission:bool=Form(False)):
    if not permission:raise HTTPException(400,'请确认有权将文档用于本地检索。')
    if not title.strip() or len(title)>200:raise HTTPException(400,'标题应为1至200字符。')
    raw=await file.read(8*1024*1024+1)
    if len(raw)>8*1024*1024:raise HTTPException(413,'文件超过8MB。')
    if not GPU.acquire(blocking=False):raise HTTPException(409,'请等待当前任务结束再导入。')
    try:
        def ingest():return SERVICE.retrieval_ready().ingest(raw,Path(file.filename or 'document.txt').name,title.strip(),source[:1000],updated[:100])
        return await asyncio.to_thread(ingest)
    except (ValueError,UnicodeError) as exc:raise HTTPException(400,str(exc))
    finally:GPU.release()


@app.get('/api/evaluation')
def evaluation():
    p=DATA/'evaluation-summary.json'
    result=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'state':'pending','note':'评测尚未完成，不显示虚构指标。'}
    latest=DATA/'current-regression-summary.json'
    if latest.exists():result['current_regression']=json.loads(latest.read_text(encoding='utf-8'))
    selected=DATA/'selection-summary.json'
    if selected.exists():result['extractive_update']=json.loads(selected.read_text(encoding='utf-8'))
    comparisons=[]
    for name in ['lf-selector-base-test.json','lf-selector-lf-v1-test.json']:
        candidate=DATA/name
        if candidate.exists():comparisons.append(json.loads(candidate.read_text(encoding='utf-8'))['summary'])
    if comparisons:result['lf_behavior_test']=comparisons
    health=DATA/'health-holdout-lf-v1.json'
    if health.exists():result['lf_health_test']=json.loads(health.read_text(encoding='utf-8'))['summary']
    second=DATA/'lf-v2-summary.json'
    if second.exists():result['lf_v2_test']=json.loads(second.read_text(encoding='utf-8'))
    alignment=DATA/'alignment-evaluation-summary.json'
    if alignment.exists():result['alignment_update']=json.loads(alignment.read_text(encoding='utf-8'))
    return result


@app.exception_handler(HTTPException)
async def error_handler(request,exc):return JSONResponse({'error':exc.detail},exc.status_code)

app.mount('/static',StaticFiles(directory=str(ROOT/'static')),name='static')

@app.get('/')
def home():return FileResponse(ROOT/'static/index.html')

if __name__=='__main__':
    import argparse,uvicorn,webbrowser
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=7860);parser.add_argument('--open',action='store_true');args=parser.parse_args()
    if args.open:threading.Timer(1,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
