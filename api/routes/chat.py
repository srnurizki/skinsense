# Import Libraries
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from api.schemas import ChatRequest
from agent.orchestrator import app as agent_app

router = APIRouter()

# SSE
async def stream_agent(message: str, history: list):
    messages = []
    for msg in history:
        if msg.get('role') == 'user':
            messages.append(HumanMessage(content=msg.get('content', '')))
        elif msg.get('role') == 'assistant':
            messages.append(AIMessage(content=msg.get('content', '')))

    messages.append(HumanMessage(content=message))
    inputs = {'messages': messages}

    try:
        async for chunk in agent_app.astream(inputs, stream_mode='updates'):
            if 'agent' in chunk:
                out_msgs = chunk['agent'].get('messages', [])
                if out_msgs:
                    msg = out_msgs[-1]
                    content = msg.content

                    if isinstance(content, list):
                        content = "".join([c.get("text", "") for c in content if isinstance(c, dict) and "text" in c])

                    if content and isinstance(content, str):
                        yield content
    except Exception as e:
        yield f"Gagal mengeksekusi agen: {str(e)}"

    yield '[DONE]'

@router.post('/')
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_agent(request.message, request.history),
        media_type='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        })