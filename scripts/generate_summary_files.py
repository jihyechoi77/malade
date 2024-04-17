from malade.omop_interactions import Interactions
from malade.drug_categories import DrugCategoriesSummary
from malade.omop import conditions, omop_drugs, printing_name
from malade.utils.formatting import format_list
from malade.utils.pydantic import load

def omop_results_summary(cache_path: str="outputs/interactions.json") -> str:
    interactions = load(Interactions, cache_path)()
    output = ""
    def addline(*items) -> None:
        nonlocal output
        output += " ".join(str(i) for i in items) + "\n"
        
    for category in omop_drugs:
        category_str = str(category.value)
        if category_str not in interactions.categories:
            continue
        category_results = interactions.categories[category_str]

        addline(f"# {printing_name(category)}")
        addline()
        for condition in conditions:
            condition_value = condition.value
            if condition_value not in category_results.conditions:
                continue
            results = category_results.conditions[condition_value]
            if not results.net_effect_computed:
                continue
                
            addline(f"## {printing_name(condition)}")
            addline()
            addline(f"**Net effect (computed)**: {results.net_effect}")
            addline()
            addline(f"**Confidence/Magnitude Measures**: Confidence {results.confidence}, probability {results.probability}, frequency {results.frequency}, evidence {results.evidence}")
            addline()
            addline("**Justification**:")
            addline(results.net_effect_justification)
            addline()
            
    return output

def generate_omop_results_summary_file(path: str = "outputs/omop_results.md"):
    with open(path, "w") as f:
        f.write(omop_results_summary())

def drug_selection_summary(cache_path: str="outputs/representative_drugs.json") -> str:
    representative_drugs = load(DrugCategoriesSummary, cache_path)()
    output = ""
    def addline(*items) -> None:
        nonlocal output
        output += " ".join(str(i) for i in items) + "\n"
        
    for category in omop_drugs:
        category_str = str(category.value)
        if category_str not in representative_drugs.summaries:
            continue
        category_results = representative_drugs.summaries[category_str]

        addline(f"# {printing_name(category)}")
        addline()
        addline(f"**Representatives**: {format_list(category_results.representative_drugs)}")
        addline()
        addline("**Justification**:")
        addline(category_results.justification)
        addline()
            
    return output

def generate_drug_selection_summary_file(path: str = "outputs/representative_drugs.md"):
    with open(path, "w") as f:
        f.write(drug_selection_summary())

generate_omop_results_summary_file()
generate_drug_selection_summary_file()
