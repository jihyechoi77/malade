from malade.omop_evaluation import *
import pandas as pd
import altair as alt
from sklearn.metrics import RocCurveDisplay, roc_curve, classification_report
from functools import partial
import re
import warnings
import logging

logging.getLogger().setLevel(logging.ERROR)

for auc_type, label in [
    ("ADE", OMOPLabels.INCREASE),
    ("Effect", OMOPLabels.NO_EFFECT),
]:
    confidence_auc = omop_auc(confidence, target_class=label)
    probability_auc = omop_auc(confidence, target_class=label)
    print(f"{auc_type} auc: confidence {confidence_auc} probability {probability_auc}")

    plot_rocs(confidence, path="img/roc_confidence.pdf")
    plot_sensitivity_specificity(confidence, path="img/sensitivity_specificity_confidence.pdf")
    plot_rocs(probability, path="img/roc_probability.pdf")
    plot_sensitivity_specificity(probability, path="img/sensitivity_specificity_probability.pdf")

# OMOP Ground Truth and Predictions
ground_truth_no_blue = plot_interactions(
    omop_table,
    width=600,
    height=400,
    legend=False,
    keep_blue=False
)
predictions = plot_interactions(
    predicted_omop_table(exclude_fn=exclude_prob_or_rare_and_weak),
    legend=False,
    height=400,
    width=600
)

chart = (ground_truth_no_blue | predictions).configure_axisX(
    orient="top",
    labelAngle=-30,
).configure_axisY(
    titlePadding=80,
).configure_axis(
    labelFontSize=25,
    titleFontSize=25,
    labelLimit=1000,
).configure_legend(
    titleFontSize=25,
    labelFontSize=25,
)

chart.save("img/omop_results.pdf")
plot_interactions(
    predicted_omop_table(exclude_fn=noop),
    save_to="img/omop_predicted_all.pdf"
)
plot_interactions(
    predicted_omop_table(exclude_fn=exclude_prob_or_rare_and_weak),
    save_to="img/omop_predicted_postprocessing.pdf"
)
plot_interactions(omop_table, save_to="img/omop_ground_truth.pdf")

# F1 scores and confusion matrix
y, yhat = to_verified_labels(predicted_omop_table(interactions, noop))
label_map = {
    OMOPLabels.DECREASE: 0,
    OMOPLabels.NO_EFFECT: 1,
    OMOPLabels.INCREASE: 2,
}
y = [label_map[yi] for yi in y]
yhat = [label_map[yi] for yi in yhat]

fig, ax = plt.subplots(figsize=(5,4.5))

omop_confusion(
    exclude_prob_or_rare_and_weak, 
    ax=ax,
    colorbar=True, 
    im_kw={"vmin": 0, "vmax": 1}
)

plt.savefig("img/confusion_matrix_postprocessing.pdf", bbox_inches='tight', pad_inches=0)


fig, ax = plt.subplots(figsize=(5,4.5))

omop_confusion(
    noop, 
    ax=ax,
    colorbar=True, 
    im_kw={"vmin": 0, "vmax": 1}
)

plt.savefig("img/confusion_matrix.pdf", bbox_inches='tight', pad_inches=0)

print("F1: no postprocessing")
print(omop_f1(noop))

print("F1: postprocessing")
print(omop_f1(exclude_prob_or_rare_and_weak))

# Critic evaluation
feedback_regex = re.compile(r'FUNC\:\s*{\n\s*"name": "feedback"')

def get_drug_log(category: OMOPDrugs) -> str:
    path = f"logs/DrugFinder-{str(category.value)}.log"
    with open(path) as f:
        return f.read()

def get_drugs(category: OMOPDrugs) -> list[str]:
    return interactions.categories[str(category.value)].representative_drugs
    
def get_drug_outcome_log(drug: str, condition: OMOPConditions) -> str:
    path = f"logs/DrugOutcomeInfoAgent-{str(condition.value)}-{drug}.log"
    with open(path) as f:
        return f.read()

def get_classification_log(category: OMOPDrugs, condition: OMOPConditions) -> str:
    path = f"logs/CategoryOutcomeRiskAgent-{str(condition.value)}-{str(category.value)}.log"
    with open(path) as f:
        return f.read()

def critic_count(text: str) -> str:
    # Occurs twice per call to critic
    return len(feedback_regex.findall(text)) // 2

def mean(vals: list[float]) -> float:
    return sum(vals)/len(vals)

drug_counts = {cat: critic_count(get_drug_log(cat)) for cat in omop_drugs}
drug_outcome_counts = {
    (cat, condition, drug): critic_count(
        get_drug_outcome_log(drug, condition)
    ) 
    for cat in omop_drugs
    for condition in conditions
    for drug in get_drugs(cat)
}
classification_counts = {
    (cat, condition): critic_count(
        get_classification_log(cat, condition)
    ) 
    for cat in omop_drugs
    for condition in conditions
}

def to_dist(lst: list[int]) -> dict[int, float]:
    counts = {}

    for v in lst:
        if v not in counts:
            counts[v] = 1
        else:
            counts[v] += 1

    return {k: v/len(lst) for k, v in counts.items()}

def table(labeled_dicts):
    outputs = {}

    for name, critic_counts in labeled_dicts:
        count_dist = to_dist(list(critic_counts.values()))
        outputs[name] = {
            "0,1": count_dist.get(0, 0) + count_dist.get(1, 0),
            "2+": sum(c for i, c in count_dist.items() if i >= 2),
        }

    return outputs

for k, v in table([
    ("DrugFinder", drug_counts),
    ("DrugOutcomeInfoAgent", drug_outcome_counts),
    ("CategoryOutcomeRiskAgent", classification_counts),
]).items():
    print(k)
    print(f"No correction rate: {v['0,1']}, correction rate {v['2+']}\n")
    
