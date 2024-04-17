from pydantic import BaseModel
from typing import Any
from enum import Enum

class DrugCategory(BaseModel):
    category: str
    subcategories: list[Any] = []

    def __str__(self):
        return self.category

class OMOPDrugs(Enum):
    ACE_INHIBITORS = "angiotensin converting enzyme inhibitor"
    AMPHOTERICIN_B = "amphotericin b"

    # Antibiotics
    ERYTHROMYCINS = "erythromycin"
    SULFONAMIDES = "sulfonamide"
    TETRACYCLINES = "tetracycline"
    ANTIBIOTICS = DrugCategory(
        category="antibiotics",
        subcategories=[ERYTHROMYCINS, SULFONAMIDES, TETRACYCLINES]
    )

    # Antiepileptics
    CARBAMAZEPINE = "carbamazepine"
    PHENYTOIN = "phenytoin"
    ANTIEPILEPTICS = DrugCategory(
        category="antiepileptics",
        subcategories=[CARBAMAZEPINE, PHENYTOIN]
    )

    BENZODIAZEPINES = "benzodiazepine"
    BETA_BLOCKERS = "beta blocker"

    # Bisphosphonates
    ALENDRONATE = "alendronate"
    BISPHOSPHONATES = DrugCategory(
        category="Bisphosphonates",
        subcategories=[ALENDRONATE]
    )

    TRICYCLICS = "tricyclic antidepressant"
    TYPICAL_ANTIPSYCHOTICS = "typical antipsychotic"
    WARFARIN = "warfarin"

def get_subcategories(category: DrugCategory):
    return [OMOPDrugs(sc) for sc in category.subcategories]

class OMOPConditions(Enum):
    ANGIOEDEMA = "angioedema"
    APLASTIC_ANEMIA = "aplastic anemia"
    ACUTE_LIVER_INJURY = "acute liver injury"
    BLEEDING = "bleeding"
    HIP_FRACTURE = "hip fracture"
    HOSPITALIZATION = "hospitalization"
    MYOCARDIAL_INFARCTION = "myocardial infarction"
    MORTALITY_AFTER_MI = "mortality after myocardial infarction"
    RENAL_FAILURE = "renal failure"
    GI_ULCER_HOSPITALIZATION = "gastrointestinal ulcer hospitalization"


# Contains subcategories
# This is used for classification
omop_drugs = [
    OMOPDrugs.ACE_INHIBITORS,
    OMOPDrugs.AMPHOTERICIN_B,
    OMOPDrugs.ERYTHROMYCINS,
    OMOPDrugs.SULFONAMIDES,
    OMOPDrugs.TETRACYCLINES,
    OMOPDrugs.CARBAMAZEPINE,
    OMOPDrugs.PHENYTOIN,
    OMOPDrugs.BENZODIAZEPINES,
    OMOPDrugs.BETA_BLOCKERS,
    OMOPDrugs.ALENDRONATE,
    OMOPDrugs.TRICYCLICS,
    OMOPDrugs.TYPICAL_ANTIPSYCHOTICS,
    OMOPDrugs.WARFARIN,
]

# This is used for evaluation (i.e. risks of a category are the
# maximal risk of a subcategory, if multiple subcategories exist)
omop_drugs_for_evaluation = [
    OMOPDrugs.ACE_INHIBITORS,
    OMOPDrugs.AMPHOTERICIN_B,
    OMOPDrugs.ANTIBIOTICS,
    OMOPDrugs.ANTIEPILEPTICS,
    OMOPDrugs.BENZODIAZEPINES,
    OMOPDrugs.BETA_BLOCKERS,
    OMOPDrugs.BISPHOSPHONATES,
    OMOPDrugs.TRICYCLICS,
    OMOPDrugs.TYPICAL_ANTIPSYCHOTICS,
    OMOPDrugs.WARFARIN,
]

conditions = [
    OMOPConditions.ANGIOEDEMA,
    OMOPConditions.APLASTIC_ANEMIA,
    OMOPConditions.ACUTE_LIVER_INJURY,
    OMOPConditions.BLEEDING,
    OMOPConditions.HIP_FRACTURE,
    OMOPConditions.HOSPITALIZATION,
    OMOPConditions.MYOCARDIAL_INFARCTION,
    OMOPConditions.MORTALITY_AFTER_MI,
    OMOPConditions.RENAL_FAILURE,
    OMOPConditions.GI_ULCER_HOSPITALIZATION,
]

class OMOPLabels(Enum):
    NO_EFFECT = "no-effect"
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_EFFECT_VERIFIED = "no-effect-verified"

def printing_name(item: OMOPDrugs | OMOPConditions | OMOPLabels) -> str:
    if isinstance(item, OMOPDrugs):
        name_map = {
            OMOPDrugs.ACE_INHIBITORS: "ACE Inhibitors",
            OMOPDrugs.AMPHOTERICIN_B: "Amphotericin B",
            OMOPDrugs.ANTIBIOTICS: "Antibiotics",
            OMOPDrugs.ANTIEPILEPTICS: "Antiepileptics",
            OMOPDrugs.BENZODIAZEPINES: "Benzodiazepines",
            OMOPDrugs.BETA_BLOCKERS: "Beta Blockers",
            OMOPDrugs.BISPHOSPHONATES: "Bisphosphonates",
            OMOPDrugs.TRICYCLICS: "Tricyclics",
            OMOPDrugs.TYPICAL_ANTIPSYCHOTICS: "Typical Antipsychotics",
            OMOPDrugs.WARFARIN: "Warfarin",
            # Subcategories
            OMOPDrugs.ERYTHROMYCINS: "Erythromycins",
            OMOPDrugs.SULFONAMIDES: "Sulfonamides",
            OMOPDrugs.TETRACYCLINES: "Tetracyclines",
            OMOPDrugs.CARBAMAZEPINE: "Carbamazepine",
            OMOPDrugs.PHENYTOIN: "Phenytoin",
            OMOPDrugs.ALENDRONATE: "Alendronate",
        }
    elif isinstance(item, OMOPConditions):
        name_map = {
            OMOPConditions.ANGIOEDEMA: "Angioedema",
            OMOPConditions.APLASTIC_ANEMIA: "Aplastic Anemia",
            OMOPConditions.ACUTE_LIVER_INJURY: "Acute Liver Injury",
            OMOPConditions.BLEEDING: "Bleeding",
            OMOPConditions.HIP_FRACTURE: "Hip Fracture",
            OMOPConditions.HOSPITALIZATION: "Hospitalization",
            OMOPConditions.MYOCARDIAL_INFARCTION: "Myocardial Infarction",
            OMOPConditions.MORTALITY_AFTER_MI: "Mortality After MI",
            OMOPConditions.RENAL_FAILURE: "Renal Failure",
            OMOPConditions.GI_ULCER_HOSPITALIZATION: "GI Ulcer Hospitalization",
        }
    else:
        name_map = {
            OMOPLabels.NO_EFFECT: "No Effect",
            OMOPLabels.INCREASE: "Increased Risk",
            OMOPLabels.DECREASE: "Decreased Risk",
            OMOPLabels.NO_EFFECT_VERIFIED: "No Effect (Evaluated)",
        }

    return name_map.get(item, str(item.value)) # type: ignore

omop_table = {
    k: {
        c: OMOPLabels.NO_EFFECT
        for c in conditions
    }
    for k in omop_drugs_for_evaluation
}

def no_effect_validated(*args: OMOPConditions) -> list[tuple[OMOPConditions, OMOPLabels]]:
    return [(a, OMOPLabels.NO_EFFECT_VERIFIED) for a in args]

omop_interactions = {
    OMOPDrugs.ACE_INHIBITORS: [(OMOPConditions.ANGIOEDEMA, OMOPLabels.INCREASE), (OMOPConditions.HOSPITALIZATION, OMOPLabels.DECREASE)] + no_effect_validated(
        OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.HIP_FRACTURE, OMOPConditions.GI_ULCER_HOSPITALIZATION,
    ),
    OMOPDrugs.AMPHOTERICIN_B: [(OMOPConditions.RENAL_FAILURE, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.ANGIOEDEMA, OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.ACUTE_LIVER_INJURY, OMOPConditions.HIP_FRACTURE, OMOPConditions.MORTALITY_AFTER_MI,
    ),
    OMOPDrugs.ANTIBIOTICS: [(OMOPConditions.ACUTE_LIVER_INJURY, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.BLEEDING, OMOPConditions.HIP_FRACTURE, OMOPConditions.MYOCARDIAL_INFARCTION, OMOPConditions.RENAL_FAILURE,
    ),
    OMOPDrugs.ANTIEPILEPTICS: [(OMOPConditions.APLASTIC_ANEMIA, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.ANGIOEDEMA, OMOPConditions.MORTALITY_AFTER_MI, OMOPConditions.RENAL_FAILURE, OMOPConditions.GI_ULCER_HOSPITALIZATION,
    ),
    OMOPDrugs.BENZODIAZEPINES: [(OMOPConditions.HIP_FRACTURE, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.ANGIOEDEMA, OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.ACUTE_LIVER_INJURY, OMOPConditions.BLEEDING, OMOPConditions.MYOCARDIAL_INFARCTION, OMOPConditions.RENAL_FAILURE,
    ),
    OMOPDrugs.BETA_BLOCKERS: [(OMOPConditions.MORTALITY_AFTER_MI, OMOPLabels.DECREASE)] + no_effect_validated(
        OMOPConditions.ANGIOEDEMA, OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.ACUTE_LIVER_INJURY, OMOPConditions.HIP_FRACTURE, OMOPConditions.RENAL_FAILURE, OMOPConditions.GI_ULCER_HOSPITALIZATION,
    ),
    OMOPDrugs.BISPHOSPHONATES: [(OMOPConditions.GI_ULCER_HOSPITALIZATION, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.ACUTE_LIVER_INJURY, OMOPConditions.MYOCARDIAL_INFARCTION, OMOPConditions.RENAL_FAILURE,
    ),
    OMOPDrugs.TRICYCLICS: [(OMOPConditions.MYOCARDIAL_INFARCTION, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.ACUTE_LIVER_INJURY, OMOPConditions.BLEEDING, OMOPConditions.RENAL_FAILURE,
    ),
    OMOPDrugs.TYPICAL_ANTIPSYCHOTICS: [(OMOPConditions.MYOCARDIAL_INFARCTION, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.RENAL_FAILURE, OMOPConditions.GI_ULCER_HOSPITALIZATION,
    ),
    OMOPDrugs.WARFARIN: [(OMOPConditions.BLEEDING, OMOPLabels.INCREASE)] + no_effect_validated(
        OMOPConditions.ANGIOEDEMA, OMOPConditions.APLASTIC_ANEMIA, OMOPConditions.HIP_FRACTURE, OMOPConditions.MORTALITY_AFTER_MI, OMOPConditions.RENAL_FAILURE,
    ),
}

for drug, updates in omop_interactions.items():
    for (condition, label) in updates:
        omop_table[drug][condition] = label
