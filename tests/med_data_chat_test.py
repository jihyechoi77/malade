import langroid.language_models as lm
from doc.document_handler import get_document_task

local = False
critic = True
optimized = True
use_fn_api = not local
if local:
    llm = lm.OpenAIGPTConfig(
        chat_model="ollama/mixtral:8x7b-instruct-v0.1-q6_K",
        chat_context_length=128000,
    )
else:
    llm=lm.OpenAIGPTConfig()



doc_task = get_document_task(
    sources=["src/doc/amiodarone.pdf"],
    collection_name="amiodarone",
    llm=llm,
    optimized=optimized,
    critic=critic,
)

doc_task.run("Could you provide a list of drugs that have potentially negative interactions with amiodarone?")
