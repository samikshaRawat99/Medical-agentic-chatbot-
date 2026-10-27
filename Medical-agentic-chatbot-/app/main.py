from __future__ import annotations
import base64
import io
import secrets
import json
from typing import Optional
from pathlib import Path
from uuid import uuid4
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
from app.config import settings
from app.agent import graph, stream_agent
from app.history import (
    list_conversations,
    get_conversation,
    save_conversation_turn,
    delete_conversation,
)

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='Medical Agent Educational API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])

@app.middleware("http")
async def add_cache_control_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    history: list[dict[str,str]] = Field(default_factory=list,max_length=10)
    image_data: Optional[str] = None
    
def auth(authorization: Optional[str]):
    token = settings.api_token.strip()
    if token and not token.startswith(('replace-with', 'replace_with')) and not secrets.compare_digest(authorization or '', f'Bearer {token}'):
        raise HTTPException(status_code=401, detail='Invalid API token')

def inspect_image(data):
    if not data:
        return ''
    try:
        header,payload=data.split(',',1)
        mime=header.split(';')[0].split(':')[1]
        raw=base64.b64decode(payload,validate=True)
        if mime not in ('image/jpeg','image/png','image/webp') or len(raw)>5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError,IndexError) as exc:
        raise HTTPException(status_code=400,detail=f'Invalid image: {exc}')
    client=Groq(api_key=settings.groq_api_key,timeout=45)
    result=client.chat.completions.create(model=settings.groq_vision_model,messages=[{'role':'user','content':[{'type':'text','text':'Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.'},{'type':'image_url','image_url':{'url':data}}]}],max_tokens=300)
    return result.choices[0].message.content or ''

@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok','configured':bool(settings.groq_api_key and settings.pinecone_api_key),'tavily':bool(settings.tavily_api_key)}

@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: Optional[str] = Header(default=None),
    conversation_id: Optional[str] = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(503,'Set Groq_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation=inspect_image(payload.image_data)

        state = graph.invoke(
            {
                'question': payload.message,
                'history': history,
                'image_description': observation,
                'kb_query': '',
                'web_query': '',
                'kb': [],
                'web': [],
                'answer': '',
                'route': '',
                'guard_passed': True,
                'guard_violations': [],
                'guard_suggestion': '',
                'revision_count': 0
            },
            config=config
        )

        sources = [{'type':'pdf','page':x['page'],'score':x['score']} for x in state.get('kb', [])] + [{'type':'web','url':x['url'],'title':x['title']} for x in state.get('web', [])]
        
        # Persist conversation turn and auto-generate topic title if first message
        title = save_conversation_turn(
            conv_id=thread_id,
            user_question=payload.message,
            bot_answer=state.get('answer', ''),
            sources=sources
        )

        return {
            'answer': state.get('answer', ''),
            'sources': sources,
            'tools': ['pinecone'] + (['tavily'] if state.get('route')=='web' and settings.tavily_api_key else []) + (['vision'] if observation else []),
            'conversation_id': thread_id,
            'title': title
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')

@app.post('/api/chat/stream')
async def chat_stream(
    payload: ChatRequest,
    authorization: Optional[str] = Header(default=None),
    conversation_id: Optional[str] = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(503, 'Set GROQ_API_KEY and PINECONE_API_KEY in .env')

    history = [x for x in payload.history if x.get('role') in ('user', 'assistant') and isinstance(x.get('content'), str)][:10]
    thread_id = conversation_id or str(uuid4())
    observation = inspect_image(payload.image_data)

    async def event_generator():
        try:
            async for raw_event in stream_agent(payload.message, history, observation):
                if raw_event.startswith("data: "):
                    try:
                        data = json.loads(raw_event[6:].strip())
                        if data.get("type") == "complete":
                            final_answer = data.get("answer", "")
                            sources = data.get("sources", [])
                            # Persist to SQLite memory and auto-generate title
                            title = save_conversation_turn(
                                conv_id=thread_id,
                                user_question=payload.message,
                                bot_answer=final_answer,
                                sources=sources
                            )
                            yield f"data: {json.dumps({'type': 'done', 'conversation_id': thread_id, 'title': title, 'sources': sources})}\n\n"
                            continue
                    except Exception:
                        pass
                yield raw_event
        except Exception as exc:
            print("Stream error:", repr(exc), flush=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get('/api/conversations')
def get_conversations(authorization: Optional[str] = Header(default=None)):
    auth(authorization)
    return list_conversations()

@app.get('/api/conversations/{conv_id}')
def get_single_conversation(conv_id: str, authorization: Optional[str] = Header(default=None)):
    auth(authorization)
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, 'Conversation not found')
    return conv

@app.delete('/api/conversations/{conv_id}')
def delete_single_conversation(conv_id: str, authorization: Optional[str] = Header(default=None)):
    auth(authorization)
    deleted = delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(404, 'Conversation not found')
    return {'status': 'deleted', 'id': conv_id}


@app.post('/api/transcribe')
async def transcribe(file:UploadFile=File(...),authorization:Optional[str]=Header(default=None)):
    auth(authorization)
    data=await file.read(10_000_001)
    if len(data)>10_000_000:
        raise HTTPException(413,'Audio exceeds 10 MB')
    audio=io.BytesIO(data); audio.name='voice.webm'
    try:
        result=Groq(api_key=settings.groq_api_key).audio.transcriptions.create(model='whisper-large-v3',file=audio)
        return {'text':result.text}
    except Exception:
        raise HTTPException(502,'Speech transcription unavailable')

@app.post('/api/speak')
def speak(payload:dict,authorization:Optional[str]=Header(default=None)):
    auth(authorization)
    return {'status':'client_tts_recommended','message':'Speech synthesis is handled directly in browser via Web Speech API.'}