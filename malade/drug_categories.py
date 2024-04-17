import os
import langroid as lr
import langroid.language_models as lm
from sqlalchemy import text
from sqlalchemy.engine import URL, create_engine
from pydantic import BaseModel
from functools import partial
from malade.critic_agent import FeedbackTool, FinalAnswerTool, CriticConfig, Critic
from malade.utils.openfda import get_drugs_of_class
from malade.utils.formatting import format_list
from malade.utils.constants import DONE
from malade.utils.task import Task
from malade.utils.pydantic import load, save
from malade.tools.passing import handle_with_pass_to
from malade.omop import omop_drugs
from fire import Fire
import logging

logger = logging.getLogger(__name__)
# set level to info
logger.setLevel(logging.INFO)

lr.agent.special.doc_chat_agent.apply_nest_asyncio()

def drug_count_query(drugs: list[str]) -> str:
    drug_pattern = lambda drug: f"'%{drug}%'"
    return f"""
    SELECT
        drug,
        COUNT(*) as prescription_count
    FROM mimiciv_hosp.prescriptions
    WHERE drug ILIKE ANY (ARRAY[{','.join(drug_pattern(d) for d in drugs)}])
    GROUP BY drug 
    ORDER BY prescription_count DESC
    """

class SubmitAnswer(lr.ToolMessage):
    request: str = "submit_answer"
    purpose: str = "To express your answer, including the representative drugs and your justification."
    representative_drugs: list[str]
    justification: str


class DrugCategorySummary(BaseModel):
    representative_drugs: list[str]
    justification: str

class DrugCategoriesSummary(BaseModel):
    summaries: dict[str, DrugCategorySummary]
    
def get_drugs_by_category(
    categories: list[str],
    db: str = "mimiciv",
    cache_path: str="outputs/representative_drugs.json",
    llm: lm.OpenAIGPTConfig = lm.AzureConfig(),
    recompute: bool = False,
) -> DrugCategoriesSummary:
    uri = URL.create(
        drivername="postgresql",
        username=os.getlogin(),
        host="localhost",
        database=db,
    )

    engine = create_engine(uri)

    load_representative_drugs = load(DrugCategoriesSummary, cache_path)
    save_representative_drugs = save(DrugCategoriesSummary, cache_path)
    
    if os.path.exists(cache_path):
        representative_drugs: DrugCategoriesSummary = load_representative_drugs()
    else:
        representative_drugs = DrugCategoriesSummary(summaries={})

    for cat in categories:
        cat_str = str(cat)
        if cat_str in representative_drugs.summaries and not recompute:
            continue

        drugs = list(get_drugs_of_class(cat))

        with engine.begin() as conn:
            counts = conn.execute(text(drug_count_query(drugs))).fetchall()
        representative_drug_extraction_prompt = f"""
            You are a helpful assistant with general medical and
            pharmacological knowledge.  I will provide you with a list
            of drugs, and the result of a query on a medical database
            with their usage rates; your goal is to find N
            representative drugs in category {cat} out of the provided
            drugs.

            Prefer generic names if possible, and do not include both
            a brand and generic name for the same drug in your list.

            If possible, prefer drugs with different active
            ingredients (i.e. avoid derivatives of a drug already in
            the list), keeping your choices to the most basic variant
            of a given drug from the list (use the total prescription
            rate of variants of the same base drug to select the top
            drugs); disregard this if you cannot find N with this
            restriction. If fewer than N meet the conditions, you may
            include fewer than N (but never more).

            The names of the selected representatives must EXACTLY
            match one of the provided drugs; choose the names from
            the original list, not the database query.

            You must provide your final answer with the `final_answer`
            tool/function; make sure to clearly state my question, as
            well as the reasoning used to derive the answer. Include
            the requirements on your answer in the `question`
            field.

            Once the critic is satisfied with your answer, send me
            the answer with the `submit_answer` tool/function.
            """

        representative_drug_agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=llm,
                system_message=representative_drug_extraction_prompt,
            )
        )
        representative_drug_agent.enable_message(SubmitAnswer)
        representative_drug_agent.enable_message(FinalAnswerTool)
        representative_drug_agent.enable_message(FeedbackTool, use=False)

        cat_summary = {}
        def handle_submit_answer(msg: SubmitAnswer) -> str:
            cat_summary["representative_drugs"] = [
                d.upper() for d in msg.representative_drugs
            ]
            cat_summary["justification"] = msg.justification
            return DONE
        setattr(representative_drug_agent, "submit_answer", handle_submit_answer)

        critic = Critic(
            CriticConfig(
                llm=llm,
                user_message=f"""
                You are also an expert in medical and pharmacological
                reasoning.

                Your goal is to ensure that the selected drugs are
                distinct members of the category {cat} of drugs.
                You will consider information provided directly
                to the user to be reliable (for example, this
                might include prescription rates and a complete
                list of drugs in category {cat}). Unless this
                contradicts your pharmacological knowledge,
                the user's choices of representatives for a
                category are acceptable unless they do not
                represent the basic form of a given drug.
                """
            )
        )
        representative_drug_task = Task(
            representative_drug_agent,
            name=f"RepresentativeDrugs-{cat}",
            interactive=False,
            erase_substeps=True,
        )
        representative_drug_task.add_sub_task(Task(
            critic,
            interactive=False,
            name="Critic",
        ))
        handle_with_pass_to(representative_drug_agent, FinalAnswerTool, "Critic")
            
        representative_drug_task.run(
            f"""
            Out of {format_list(drugs)}, which three are the most common?

            The result of a query on a medical database is below:
            {counts}
            """
        )
        representative_drugs.summaries[cat_str] = DrugCategorySummary.validate(
            cat_summary
        )
        save_representative_drugs(representative_drugs)

    return representative_drugs

if __name__ == "__main__":
    omop_drug_values: list[str] = list(map(lambda v: v.value, omop_drugs)) # type: ignore
    Fire(partial(get_drugs_by_category, omop_drug_values))
