from fastapi import FastAPI

from transformers import pipeline

from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from pydantic import BaseModel

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(
    title="API de resolucion de preguntas",
    description="Esta API es la encargada de responder preguntas dado un contexto",
    version = "1.0",
    middleware=middleware
)

class request_model(BaseModel):

    pregunta: str 

    class Config:
        
        schema_extra = {
            "example": {
                "pregunta": "Que acciones son requeridad siguiendo el lineamiento de ayudar los clientes en la transición hacia un futuro sostenible"
            }
        }

translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
answer_generator = pipeline("question-answering")

def respuesta_pregunta_con_contexto(pregunta_esp, contexto_eng):
    
    pregunta_eng = translator_es_en(pregunta_esp)[0]['translation_text']
    respuesta_eng = answer_generator(question=pregunta_eng, context=contexto_eng)['answer']
    respuesta_esp = translator_en_es(respuesta_eng)[0]['translation_text']
    
    return respuesta_esp
    

with open('context/colombia_contexto.txt') as f:
    colombia_context = ''.join(f.readlines()).replace('\n', ' ')

with open('context/espana_contexto.txt') as f:
    espana_contexto = ''.join(f.readlines()).replace('\n', ' ')

with open('context/mexico_contexto.txt') as f:
    mexico_contexto = ''.join(f.readlines()).replace('\n', ' ')


@app.post('/pregunta_colombia')
def pregunta_colombia(input: request_model):

    return respuesta_pregunta_con_contexto(input.pregunta,colombia_context)

@app.post('/pregunta_espana')
def pregunta_espana(input: request_model):

    return respuesta_pregunta_con_contexto(input.pregunta,espana_contexto)

@app.post('/pregunta_mexico')
def pregunta_mexico(input: request_model):

    return respuesta_pregunta_con_contexto(input.pregunta,mexico_contexto)

