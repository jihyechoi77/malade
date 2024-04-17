from langroid.utils.pydantic_utils import temp_params
from rich.prompt import Prompt
import langroid as lr
import langroid.language_models as lm
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from malade.doc.doc_chat_config import doc_chat_config_dict
from malade.utils.openfda import get_drug_label_details, extract_html_text, get_drugs_of_class
from malade.utils.formatting import format_list
from langroid.agent.tool_message import ToolMessage
from langroid.agent.chat_agent import ChatAgent, ChatDocument
from langroid.mytypes import Document
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)
from malade.utils.task import Task
from malade.utils.constants import DONE
from langroid.utils.constants import NO_ANSWER
from langroid.utils.output import status
from typing import Optional
import logging

logger = logging.getLogger(__name__)
# set level to info
logger.setLevel(logging.INFO)

class DrugMetadata(lr.DocMetaData):
    drug: str
    label_section: str

class DrugDoc(lr.Document):
    metadata: DrugMetadata
    doc_type: str = "drug_doc"

class CategoryMetadata(lr.DocMetaData):
    category: str
    sub_categories: list[str]

class CategoryDoc(lr.Document):
    metadata: CategoryMetadata
    doc_type: str = "category_doc"

class RelevantExtractsTool(ToolMessage):
    request: str = "relevant_extracts"
    purpose: str = "Get docs/extracts relevant to the <query>"
    query: str
    filter_drugs: list[str] = []

    @classmethod
    def examples(cls) -> list["ToolMessage"]:
        return [
            cls(
                query="Does Aspirin have interactions with Paxlovid?",
                filter_drugs=["Asprin", "Paxlovid"],
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL query in the `query`
        field.
        """


class RelevantSearchExtractsTool(ToolMessage):
    request = "relevant_search_extracts"
    purpose = """
        Get docs/extracts relevant to the <query> for a <drug> from a web search.
        """
    query: str
    drug: str

    @classmethod
    def examples(cls) -> list["ToolMessage"]:
        return [
            cls(
                query="Does Aspirin cause hallucinations?",
                drug="aspirin",
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL query in the `query` field,
        and an ACTUAL drug in the `drug` field.
        """

class DrugCategorySearchTool(ToolMessage):
    request = "drug_category_search"
    purpose = """
        Get the drugs belonging to a list of categories and get
        relevant docs from a web search.
        """
    category: str
    sub_categories: list[str]

    @classmethod
    def examples(cls) -> list["ToolMessage"]:
        return [
            cls(
                category = "antibiotics",
                sub_categories = ["erythromycins", "sulfonamides", "tetracyclines"]
            ),
        ]

    @classmethod
    def instructions(cls) -> str:
        return """
        IMPORTANT: You must include an ACTUAL category of drugs in the
        `category` field, and all sub-categories to be considered (if
        any) in the sub_categories field.
        """


class SearchDocChatAgent(DocChatAgent):
    tried_vecdb: bool = False
    tried_search: bool = False

    def __init__(self, config: DocChatAgentConfig):
        super().__init__(config)
        self.drugs_ingested = set()
        self.err_count = 0
        self.max_sequential_errs = 3

    def reset(self):
        self.tried_search = False
        self.tried_vecdb = False

    def process_dict(self, drug: str, d: dict[str, list[str]]) -> list[Document]:
        """Process a dictionary of drug label info for a specific drug
        into a list of Documents; the dictionary looks like:
        {
            "interactions": ["blah blah", "blah", ...],
            "side_effects": ["xyz", "abc", "def", ...],
            ...
        }
        Some of the texts can be very long, so they would need to be chunked,
        but to ensure recall/precision, we want to prepend the drug name and field name
        to each chunk, e.g. "Aspirin: interactions: blah blah" etc, so that
        at query time we are able to find the appropriate chunks.
        """
        drug = drug.upper()
        docs = []
        for field, text_list in d.items():
            for text in text_list:
                if "table" in field:
                    # extract tabular data as clean text
                    # hopefully it preserves some structure
                    text = extract_html_text(text) or text
                    if text is None:
                        continue
                doc = DrugDoc(
                    content=text,
                    metadata=DrugMetadata(
                        source=f"{drug} label",
                        label_section=field,
                        drug=drug,
                    ),
                )
                # prepend drug, field to each chunk to improve precision, recall
                docs.extend([
                    DrugDoc(
                        content=f"{drug}: {field}: {d.content}",
                        metadata=d.metadata,
                    )
                    for d in self.parser.split([doc])
                ])

        return docs

    def ingest_dict(self, drug: str, d: dict[str, list[str]]) -> None:
        to_ingest = self.process_dict(drug, d)
        self.ingest_docs(to_ingest, split=False)
        logger.info(f"Ingested {len(to_ingest)} chunks for {drug}")


    def llm_response(
        self,
        query: None | str | ChatDocument = None,
    ) -> ChatDocument | None:
        # instead of using DocChatAgent llm_response,
        # use the default ChatAgent llm_response,
        # since we are creating the query ourselves here, with relevant chunks etc.
        response = ChatAgent.llm_response(self, query)

        if (
            not self.tried_search and
            DONE in response.content and
            NO_ANSWER in response.content
        ):
            # snuff out a premature DONE in case search has not been tried.
            response.content = response.content.replace(DONE, "")
        return response

    async def llm_response_async(
        self,
        query: None | str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        response = await ChatAgent.llm_response_async(self, query)

        if (
            not self.tried_search and
            DONE in response.content and
            NO_ANSWER in response.content
        ):
            # snuff out a premature DONE in case search has not been tried.
            response.content = response.content.replace(DONE, "")
        return response

    def filter_drugs(self, drugs: list[str]) -> str:
        """Construct a filter which finds DrugDocs from the labels of the drugs in `drugs`"""
        return str(Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value="drug_doc"),
                )
            ],
            should=[
                FieldCondition(
                    key="metadata.drug",
                    match=MatchValue(value=d.upper()),
                )
                for d in drugs
            ],
        ).json())


    def relevant_extracts(self, msg: RelevantExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from vecdb"""
        self.tried_vecdb = True
        # We have made a function call, so reset the counter
        self.err_count = 0
        query = msg.query

        filter_docs = self.config.filter
        msg.filter_drugs = [d.upper() for d in msg.filter_drugs]

        filter_docs = self.config.filter
        if len(msg.filter_drugs) > 0:
            filter_docs = self.filter_drugs(msg.filter_drugs)
                    
        with temp_params(self.config, "filter", filter_docs):
            _, extracts = self.get_relevant_extracts(query)

        # Filter extracts found by search methods which
        # do not consider metadata (e.g. fuzzy search) on condition
        if len(msg.filter_drugs) > 0:
            extracts = [
                e
                for e in extracts
                if hasattr(e.metadata, "drug") and e.metadata.drug in msg.filter_drugs
            ]

        # Get the drugs from the metadata, so we don't re-ingest them
        # This is a proxy for expressly searching the whole db to get all drugs.
        self.drugs_ingested.update(
            set(d.metadata.drug.upper() for d in extracts if hasattr(d.metadata, "drug"))
        )
        if len(extracts) == 0:
            return """
            No extracts found! You can try doing a web search with the
            `relevant_search_extracts` or `drug_category_search`
            tools/function-calls.
            """
        extracts_list = "\n".join(str(e) for e in extracts)
        if self.tried_search:
            return extracts_list
        return f"""
        Below are some relevant extracts for your query:
        -----
        {extracts_list}
        -----
        If NONE of the extracts are relevant, try doing a web search with the
        `relevant_search_extracts` tool/function-call
        """

    def relevant_search_extracts(self, msg: RelevantSearchExtractsTool) -> str:
        """Get docs/extracts relevant to the query, from a web search"""
        msg.drug = msg.drug.upper()
        # We have made a function call, so reset the counter
        self.err_count = 0

        if not self.tried_vecdb:
            vec_search_results = self.relevant_extracts(
                RelevantExtractsTool(
                    query=msg.query,
                    filter_drugs=[msg.drug],
                )
            )

            if "No extracts found!" not in vec_search_results:
                return vec_search_results

        self.tried_vecdb = False
        self.tried_search = True
        query = msg.query
        drug = msg.drug
        # if the tool is trying to re-ingest a previously ingested drug,
        # we abort this since we know there will be no relevant results
        if drug in self.drugs_ingested:
            return f"No relevant extracts found for {query}"
        with status(f"Getting drug label details for {drug}..."):
            results = get_drug_label_details(drug)
        # we get a list of dicts, but we only expect one dict, so take the first one
        with status(f"Ingesting drug label details for {drug}..."):
            if len(results) > 0:
                self.ingest_dict(drug, results[0])
        self.drugs_ingested.add(drug)
        with temp_params(self.config, "filter", self.filter_drugs([drug])):
            _, extracts = self.get_relevant_extracts(query)
        return "\n".join(str(e) for e in extracts)

    def drug_category_search(self, msg: DrugCategorySearchTool) -> str:
        """Get the drugs in a category from a web search."""
        # We have made a function call, so reset the counter
        self.err_count = 0

        category = msg.category
        sub_categories = msg.sub_categories
        if len(sub_categories) == 0:
            sc_descr = ""
        else:
            sc_descr = f"with sub-categories {format_list(sub_categories)}"

        cat_descr = f"category {category} {sc_descr}"

        if not self.tried_vecdb:
            filter_docs = str(Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value="category_doc"),
                    ),
                    FieldCondition(
                        key="metadata.category",
                        match=MatchValue(value=msg.category),
                    ),
                ]
            ).json())

            with temp_params(self.config, "filter", filter_docs):
                vecdb_search_results = self.relevant_extracts(RelevantExtractsTool(
                    query=f"Which drugs are in {cat_descr}?"
                ))

            if category in vecdb_search_results:
                return vecdb_search_results

        self.tried_vecdb = False
        self.tried_search = True

        sub_categories.append(category)

        drugs: list[str] = list(set().union(*(get_drugs_of_class(sc) for sc in sub_categories)))
        drugs = [d.upper() for d in drugs]

        summary = f"""
        The drugs in {cat_descr} include:
        {format_list(drugs)}.
        """

        docs: list[Document] = [
            CategoryDoc(
                content=summary,
                metadata=CategoryMetadata(
                    category=category,
                    sub_categories=sub_categories[:-1],
                ),
            )
        ]

        # Ingest all associated drug labels, if necessary
        for drug in drugs:
            drug = drug.upper()
            extracts = self.get_relevant_chunks(drug)
            # In all cases, we have attempted to ingest the drug info
            self.drugs_ingested.add(drug)
            if drug not in set(d.metadata.drug for d in extracts if isinstance(d, DrugDoc)):
                with status(f"Processing drug label details for {drug}..."):
                    results = get_drug_label_details(drug)
                    if len(results) > 0:
                        docs.extend(self.process_dict(drug, results[0]))
                        

        self.ingest_docs(docs, split=False)
        logger.info(
            f"Ingested drug labels for {cat_descr} in {len(docs)} chunks."
        )

        return summary

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            if not self.tried_search and NO_ANSWER in msg.content:
                return f"""
                Try a web search using the `relevant_search_extracts` tool,
                to see if it can help answer the question.
                """

            if DONE not in msg.content:
                self.err_count += 1

                if self.err_count > self.max_sequential_errs:
                    return f"{DONE} {NO_ANSWER}"
                return f"""
                Do one of the following:
                1. Search for relevant text using one of the following tools/functions:
                    `relevant_extracts`, `relevant_search_extracts`, and `drug_category_search`
                2. If you could not find an answer, state that you are done and {NO_ANSWER}
                    as usual
                3. If you have found an answer, state that you are done and express the answer
                    as usual
                4. If you have no clear remaining question to answer, answer the original
                    question as in #3, and if there was no original question, respond
                    as in #2

                If you have been getting or sending repeated messages, always exit by
                stating that you are done (using the usual code) and say {NO_ANSWER}.

                DO NOT simply acknowldedge these instructions. If you have nothing
                else to say, you are done and you MUST EXIT IMMEDIATELY by stating that you are done
                with the usual code (you MUST choose #2 or #3, and #4 specifies how to choose).
                THIS IS VERY IMPORTANT. DO THIS IMMEDIATELY BEFORE WAITING FOR ANY FURTHER MESSAGES.
                """

def get_fda_handler(
        llm: lm.OpenAIGPTConfig=lm.OpenAIGPTConfig(),
        collection_name: str="malade",
        storage_path: str = ".qdrant/",
        embed: str = "openai",
        recipient: Optional[str] = "LLM",
        assistant_mode: bool = True,
) -> Task:
    config = DocChatAgentConfig(
        system_message=f"""
        You will try your best to answer my questions, in this order of preference:
        
        1. Ask me for some relevant text, and I will send you. Use the
            `relevant_extracts` tool/function-call for this
            purpose. Once you receive the text, you can use it to
            answer my question. If the question asks for information
            about a specific drug, make sure to begin by including that drug in
            the `filter_drugs` field.  If I say {NO_ANSWER}, it means
            I found no relevant docs, and you can try the next step,
            using a web search.
        2. If you are still unable to answer, you can use the `relevant_search_extracts`
           tool/function-call to get some text from a web search. Once you receive the
           text, you can use it to answer my question. If you need to identify the drugs
           in a category, use the `drug_category_search` tool/function-call instead.
        3. If you are still unable to answer, and used `filter_drugs` in your initial
            attempt with `relevant_extracts`, try again without a filter.
        4. If you still can't answer, simply say {DONE} {NO_ANSWER} 

        If given a question asking about a drug "X and Y", this is a
        combination drug, so your initial searches should be for "X and Y" not
        "X" or "Y".

        If asked a question about drugs in broad category, make to consider EVERY drug
        in the category, and in particular, if the question asks for which drugs
        in the category something is true, make CERTAIN that your answer correctly
        lists ALL drugs in the category where the condition holds.

        IMPORTANT: some fields in the FDA label data retrieved by `relevant_search_extracts`
        and `relevant_extracts` have the level of reliability of information specified
        prior to it (for example, statements of the level of reliability may precede each
        section of adverse reactions, the immediately preceding such statement is
        the one that corresponds to any given reported interaction). Make certain that your answer
        reflects the specified level of reliability. Similarly, when asked about the effect of
        a drug on a condition, ALWAYS express the magitude of the effect (i.e. how frequently
        the drug results in the condition or how frequently the drug improves the condition);
        whenever possible, make sure to explicitly state whether a condition is rarely or commonly reported.
        
        ANSWER FORMAT:
        
        ALWAYS present your answer in one of the below 2 formats:
        
        1. In case you COULD NOT find an answer:
        
        {DONE} {NO_ANSWER}  
        
        2. In case you ARE able to find an answer: 
        
        {DONE}
        ANSWER: [Your concise answer, with a brief summary of necessary context.
                 ALWAYS clarify the level of reliability of the information, if
                 specified in the extracts. If applicable, ALWAYS express the
                 magnitude of any increase or decrease in risk and any associated
                 information.]
        SOURCE: aspirin label
        EXTRACT_START_END: Aspirin can cause ... with any medicine.
        
                
        For the EXTRACT_START_END, ONLY show up to first 3 words, and last 3 words.
        """,
        n_query_rephrases=0,
        hypothetical_answer=False,
        # how many sentences in each segment, for relevance-extraction:
        # increase this if you find that relevance extraction is losing context
        # extraction_granularity=3,
        assistant_mode=assistant_mode,
        split=False,
        **doc_chat_config_dict(
            llm=llm,
            collection_name=collection_name,
            replace_collection=False,
            storage_path=f"{storage_path}/{collection_name}",
            vecdb=lr.vector_store.QdrantDBConfig,
            enable_relevance_extractor=False,
            embed=embed,
        )
    )

    config.vecdb.cloud = True # supports concurrent access

    agent = SearchDocChatAgent(config)
    agent.enable_message(RelevantExtractsTool)
    agent.enable_message(RelevantSearchExtractsTool)
    agent.enable_message(DrugCategorySearchTool)
    agent.setup_documents(filter=agent.config.filter)

    task = Task(
        agent,
        name="FDAHandler",
        interactive=False,
        llm_delegate=True,
        single_round=False,
        recipient=recipient,
    )
    return task

if __name__ == "__main__":
    task = get_fda_handler()
    while True:
        query = Prompt.ask("[blue]How can I help?")
        if query in ["x", "q"]:
            break
        task.agent.reset()
        task.run(query)
