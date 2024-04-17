import os
from typing import TypedDict, Optional
import langroid.agent.special as special
import langroid as lr
import langroid.parsing.parser as parser
import langroid.vector_store as vs
import langroid.language_models as lm
import langroid.embedding_models as em

class DocChatCommonConfig(TypedDict):
    llm: lm.OpenAIGPTConfig
    use_functions_api: bool
    use_tools: bool
    n_neighbor_chunks: int
    vecdb: vs.base.VectorStoreConfig
    parsing: parser.ParsingConfig
    relevance_extractor_config: Optional[special.RelevanceExtractorAgentConfig]

def doc_chat_config_dict(
        llm: lm.OpenAIGPTConfig,
        collection_name: str,
        replace_collection: bool = True,
        storage_path: str = ".lancedb/data/",
        embed: str = "openai", 
        enable_relevance_extractor: bool = False,
        vecdb: type[vs.base.VectorStoreConfig] = vs.LanceDBConfig
) -> DocChatCommonConfig:
    local_prefixes = ["local/", "litellm/", "ollama/"]
    local = any(
        llm.chat_model.startswith(prefix)
        for prefix in local_prefixes
    )
    use_fn_api = not local

    if embed == "openai":
        embed_cfg = lr.embedding_models.OpenAIEmbeddingsConfig()
    else:
        embed_cfg = em.RemoteEmbeddingsConfig(
            model_type="sentence-transformer",
            model_name=embed,
        )

    return {
        "llm": llm,
        "use_functions_api": use_fn_api,
        "use_tools": not use_fn_api,
        "vecdb": vecdb(
            collection_name=collection_name,
            replace_collection=replace_collection,
            storage_path=storage_path,
            embedding=embed_cfg,
        ),
        "n_neighbor_chunks": 2,
        "parsing": parser.ParsingConfig(
            splitter=parser.Splitter.TOKENS,
            chunk_size=200,
            overlap=50,
            max_chunks=10_000,
            min_chunk_chars=100,
            discard_chunk_chars=5,
            n_similar_docs=5,
            n_neighbor_ids=5, 
            pdf=parser.PdfParsingConfig(
                library="pdfplumber",
            ),
        ),
        # The below are for performance:
        # This relevance extraction step improves performance on OpenAI
        # but significantly worsens it locally
        "relevance_extractor_config": None if not enable_relevance_extractor else special.RelevanceExtractorAgentConfig(
                llm=llm,
            ),
    }
    
    
