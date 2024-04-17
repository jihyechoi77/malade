import os
import langroid as lr
import langroid.language_models as lm
from typing import Literal, Callable
from pydantic import BaseModel
from langroid.utils.constants import NO_ANSWER
from malade.utils.formatting import format_list
from malade.doc.fda_handler import get_fda_handler
from malade.utils.constants import DONE
from malade.utils.task import Task
from langroid.agent.batch import run_batch_task_gen
from malade.tools.passing import handle_with_pass_to,override_fallback_recipient_tool
from malade.tools.handling import handle_with
from malade.utils.pydantic import load, save
from malade.critic_agent import (
    FeedbackTool,
    FinalAnswerTool,
    CriticConfig,
    Critic
)
from fire import Fire
import logging
from langroid.utils.logging import setup_colored_logging
from malade.omop import omop_drugs, conditions
from malade.drug_categories import get_drugs_by_category

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

setup_colored_logging()

lr.agent.special.doc_chat_agent.apply_nest_asyncio()

omop_drug_values: list[str] = list(map(lambda v: v.value, omop_drugs)) # type: ignore
condition_values = list(map(lambda v: v.value, conditions))

class DrugCategoryConditionInteractions(BaseModel):
    raw_outputs: str = ""
    net_effect_computed: bool = False
    net_effect: Literal["increase", "decrease", "no-effect"] = "no-effect"
    net_effect_justification: str = ""
    confidence: float = 0.0
    probability: float = 0.0
    frequency: Literal["none", "rare", "common"] = "none"
    evidence: Literal["none", "weak", "strong"] = "none"

class DrugCategoryInteractions(BaseModel):
    conditions: dict[str, DrugCategoryConditionInteractions] = {}
    representative_drugs: list[str] = []

class Interactions(BaseModel):
    categories: dict[str, DrugCategoryInteractions] = {}
    
def gen_drug_condition_interaction_task(
        name: str,
        use_fn_api: bool,
        llm: lr.language_models.OpenAIGPTConfig,
        embed: str,
) -> Task:
    orchestrator_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm,
            system_message=f"""
            You will recieve questions involving medical data.  You
            are experienced in general medical reasoning, but must
            consult references for any specific medical knowledge
            required to answer my questions.

            You have access to `FDAHandler`, who will answer questions
            you ask about specific drugs using FDA data. You must use
            the `recipient_message` tool/function to ask these
            questions, and the `intended_recipient` MUST be
            `FDAHandler` anytime you use this tool. Ensure that
            you ask FDAHandler for the specific information
            you need.

            As some potential complications are listed in FDA labels
            as lacking a verified causal relationship, make certain
            that your final answer expresses the degree of reliability
            of your answer. Similarly, make sure to clearly express
            the degree of risk associated (i.e. is the condition a
            rare or a common side effect, or does a drug rarely or
            frequently result in reduced risk of a condition).

            If FDAHandler cannot answer your question then your answer
            should be {NO_ANSWER}, because the FDA label data does not
            specify the answer. If FDAHandler answers with {NO_ANSWER}
            that means that the FDA label for the drug does not
            contain the information requested (and, in particular, it
            means that it does not mention the condition); hence, your
            answer should be {NO_ANSWER}. This indicates that there
            may not be any effect on the risk of the condition, make
            sure to explain this in your justification.

            IMPORTANT: if multiple attempts fail to retrieve any
            relevant information, there is no need to continue asking
            questions to FDAHandler; assume that the information is
            not in the FDA labels and so FDAHandler cannot answer.

            You MUST specifically tell the critic why you could not
            find an answer to the question; be sure to specify that
            the FDAHandler answered with {NO_ANSWER} if that is the
            reason.

            You must provide your final answer with the `final_answer`
            tool/function; make sure to clearly state my question, the
            reasoning used to derive the answer, including the
            questions asked to FDAHandler and a summary of the
            results, as well as your final answer in the `answer`
            field.

            Once the critic is satisfied with your answer, say {DONE},
            and give me the answer and justification for it. Make sure
            to provide your answer again, do not just use the answer
            sent to the critic. Include any relevant details provided
            by FDAAgent.

            If the critic is satisfied and your answer is {NO_ANSWER},
            say {DONE} {NO_ANSWER} and provide a justification.
            IMPORTANT: say {DONE} specifically, not DONE.
            """,
        )
    )
    orchestrator_agent.enable_message(FinalAnswerTool)
    orchestrator_agent.enable_message(override_fallback_recipient_tool(
        f"""
        Do one of the following:

        1. If you intend to ask a question to FDAAgent:
            Please use the 'add_recipient' tool/function-call
            and specify FDAAgent as your `intended_recipient`.
            DO NOT REPEAT your original message; ONLY specify the 
            `intended_recipient` via this tool/function-call.
        2. If you are ready to have the critic check your answer,
            do so with the `final_answer` tool/function, as I have requested.
        3. If the critic is satisfied and you intend to submit your final answer,
            answer as I have requested, making sure to mark that you are done as usual.

        When you have decided and stated your answer (or lack of one), you must
        ALWAYS specify that you are done using the usual code.

        DO NOT simply acknowldedge these instructions. If you have nothing
        else to say, you MUST EXIT IMMEDIATELY by stating that you are done
        with the usual code (as in #3).
        """
    ))
    orchestrator_agent.enable_message(FeedbackTool, handle=True, use=False)

    task = Task(
        orchestrator_agent,
        name= f"DrugOutcomeInfoAgent-{name}",
        interactive=False,
        erase_substeps=True,
    )

    critic_agent = Critic(
        CriticConfig(
            user_message=f"""
            You are also experienced in medical reasoning, and
            have general medical knowledge. Unless the responses
            are inconsistent with your medical (or common-sense)
            knowledge, you generally trust responses from
            FDAHandler.

            The answer should express the strength of evidence for
            the answer and the magnitude of the effect. If the
            user states that FDAAgent does not have this information,
            you should accept it.

            If the answer given contains {NO_ANSWER}, accept it as 
            long as the answer clearly expresses why it was not
            possible to answer the question. If it states that
            this is because FDAHandler responded with {NO_ANSWER},
            you should accept it as sufficient justification.
            Otherwise, ask the user to express whether FDAHandler
            responded with {NO_ANSWER}, and, if not, to state
            why it was not possible to answer the question. If
            it does so, the answer is acceptable and the other
            requirements need not be enforced.
            """,
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
        )
    )
    critic_task = Task(critic_agent, name="Critic", interactive=False, recipient="AGENT")
    handle_with_pass_to(orchestrator_agent, FinalAnswerTool, "Critic")

    doc_task = get_fda_handler(
        llm=llm,
        embed=embed,
        recipient="LLM",
    )

    task.add_sub_task([doc_task, critic_task])

    return task


class CategoryEffectTool(lr.ToolMessage):
    request: str = "category_effect_tool"
    purpose: str = "To express the association of a category of drugs with a condition."
    label: Literal["increase", "decrease", "no-effect"]
    confidence: float
    probability: float
    frequency: Literal["none", "rare", "common"]
    evidence: Literal["strong", "weak"]
    justification: str


def category_effect_task(
    use_fn_api: bool,
    llm: lr.language_models.OpenAIGPTConfig,
    condition: str,
    cat_name: str,
    handler: Callable[[CategoryEffectTool], None],
) -> Task:
    omop_label_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm,
            system_message=f"""
            You are experienced in general medical reasoning and have
            general medical knowledge.

            You will be provided a list of passages answering, for
            each of a set of drugs X, whether drug X increases or
            decreases the risk of {condition}. They all belong to
            category {cat_name}.

            You must provide your final answer with the `final_answer`
            tool/function; make sure to clearly state my question, the
            reasoning used to derive the answer, including the
            evidence from the passages, as well as your final answer
            in the `answer` field.

            Once the critic is satisfied, submit your answer with the
            `category_effect` tool, making sure that the answer,
            `label`, is one of the following: "increase," "decrease,"
            or "no-effect," and make sure to include your
            justification. DO NOT use this tool before you have used
            the `final_answer` tool and have had your answer accepted
            by the critic.

            Your `justification` must clearly express the magnitude of
            risk indicated and the strength of evidence. Provide a
            `confidence` value between 0 and 1 indicating the
            confidence in your assigned `label` and a `probability`
            value indicating the probability that the drug will
            cause the condition (or prevent the condition) in a
            given patient.

            Express the frequency that the drug has an effect as
            either "none," "rare," or "common" with the `frequency`
            field and express the strength of `evidence` as either
            "strong" (for example, evidence is strong when shown in a
            clinical trial) or "weak" (for example, this applies to
            purely correlational evidence) or "none" if no evidence
            exists.
            """,
        ),
    )
    omop_label_agent.enable_message(FinalAnswerTool)
    handle_with_pass_to(omop_label_agent, FinalAnswerTool, "Critic")
    omop_label_agent.enable_message(FeedbackTool, handle=True, use=False)
    omop_label_agent.enable_message(CategoryEffectTool)

    def handle(msg: CategoryEffectTool) -> str:
        handler(msg)
        return DONE

    handle_with(omop_label_agent, CategoryEffectTool, handle)

    critic_agent = Critic(
        CriticConfig(
            user_message=f"""
            You are also experienced in medical reasoning, and have
            general medical knowledge. Unless the responses are
            inconsistent with your medical (or common-sense)
            knowledge, you generally trust responses from
            FDAHandler. Similarly, you trust that the user's
            statements about passages are correct without the need to
            review them directly.

            The answer provided should indicate an increase, decrease,
            or no effect on the risk, and must be no effect if no
            evidence linking the drug category to the risk of the
            condition exists.

            The answer should be drawn from the specified passages,
            hence, the absence of information related to a condition
            in the FDA data for all drugs in a category should be
            enough to conclude that there is no effect for that drug.

            The answer should express the degree of certainty and the
            magnitude of change in risk, ensure that the provided
            answer is consistent with the evidence.
            """,
            llm=llm,
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
        )
    )
    omop_label_task = Task(
        omop_label_agent,
        interactive=False,
        name=f"CategoryOutcomeRiskAgent-{condition}-{cat_name}"
    )
    omop_label_task.add_sub_task(Task(
        critic_agent,
        interactive=False,
        name="Critic",
    ))

    return omop_label_task


def get_omop_interactions(
    local: bool = False,
    local_model: str = "ollama/mixtral:8x7b-instruct-v0.1-q6_K",
    local_embed: str = "BAAI/bge-large-en-v1.5",
    strong_llm: Literal["azure", "openai"] = "azure",
    interactions_path: str = "outputs/interactions.json",
    recompute_interactions: bool = False,
    recompute_labels: bool = False,
) -> None:
    if local:
        llm = lr.language_models.OpenAIGPTConfig(
            chat_model=local_model,
            chat_context_length=128000,
        )
        embed = local_embed
    else:
        embed = "openai"
        if strong_llm == "azure":
            llm=lm.AzureConfig()
        else:
            llm = lm.OpenAIGPTConfig()

    use_fn_api = not local

    load_interactions = load(Interactions, interactions_path)
    save_interactions = save(Interactions, interactions_path)

    if os.path.exists(interactions_path):
        interactions: Interactions = load_interactions()
    else:
        interactions = Interactions()

    categories = get_drugs_by_category(categories=omop_drug_values, llm=llm)
    for cat in omop_drug_values:
        cat_name = str(cat)
        representative_drugs_in_cat = categories.summaries[cat_name].representative_drugs

        if cat_name not in interactions.categories:
            interactions.categories[cat_name] = DrugCategoryInteractions(
                representative_drugs=representative_drugs_in_cat,
            )

        cat_interactions = interactions.categories[cat_name]

        def gen_interaction_inputs(msg: str) -> list[str]:
            return [f"Does {drug} {msg}?" for drug in representative_drugs_in_cat]

        def merge_outputs(outputs: list[str]) -> str:
            return format_list(
                [f"Drug {drug}: {answer}" for drug, answer in zip(representative_drugs_in_cat, outputs)],
                numbered=True,
            )

        for condition in condition_values:
            def gen_task(i: int) -> Task:
                return gen_drug_condition_interaction_task(
                    llm=llm,
                    embed=embed,
                    name = f"{condition}-{representative_drugs_in_cat[i]}",
                    use_fn_api=use_fn_api,
                )

            if condition not in cat_interactions.conditions or recompute_interactions:
                logger.info(f"Retrieving FDA label information related to {cat_name} and {condition}.")
                outputs = merge_outputs(
                    run_batch_task_gen(
                        gen_task,
                        gen_interaction_inputs(f"increase or decrease the risk of {condition}"),
                        output_map=lambda doc: doc.content if doc is not None else NO_ANSWER,
                        sequential=False,
                        batch_size=5,
                    )
                )

                if condition in cat_interactions.conditions:
                    cat_interactions.conditions[condition].raw_outputs = outputs
                else:
                    cat_interactions.conditions[condition] = DrugCategoryConditionInteractions(
                        raw_outputs = outputs,
                    )
                save_interactions(interactions)

        to_update = condition_values if recompute_labels else [
            c
            for c in condition_values
            if not cat_interactions.conditions[c].net_effect_computed
        ]

        def gen_task(i: int):
            condition = to_update[i]
            def handler(msg: CategoryEffectTool) -> None:
                cat_interactions.conditions[condition].net_effect = msg.label
                cat_interactions.conditions[condition].confidence = msg.confidence
                cat_interactions.conditions[condition].probability = msg.probability
                cat_interactions.conditions[condition].frequency = msg.frequency
                cat_interactions.conditions[condition].evidence = msg.evidence
                cat_interactions.conditions[condition].net_effect_justification = msg.justification
                cat_interactions.conditions[condition].net_effect_computed = True
                save_interactions(interactions)

            return category_effect_task(
                use_fn_api=use_fn_api,
                llm=llm,
                cat_name=cat_name,
                condition=condition,
                handler=handler,
            )

        def gen_inputs(i: int):
            condition = to_update[i]
            outputs = cat_interactions.conditions[condition].raw_outputs
            return f"""
                Passages:
                {outputs}
                ---------
                Does the {cat_name} category of drugs increase the risk of
                {condition}, decrease it, or is there no clear effect?
            """

        run_batch_task_gen(
            gen_task,
            [gen_inputs(i) for i in range(len(to_update))],
            sequential=False,
            batch_size=3,
        )

if __name__ == "__main__":
    Fire(get_omop_interactions)

    
    
