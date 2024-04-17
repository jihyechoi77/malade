from malade.omop_interactions import Interactions, DrugCategoryConditionInteractions
from malade.utils.pydantic import load
from collections import defaultdict
from malade.omop import (
    omop_drugs,
    omop_drugs_for_evaluation,
    conditions,
    omop_table,
    OMOPDrugs,
    OMOPConditions,
    OMOPLabels,
    DrugCategory,
    printing_name,
)
from typing import Iterable, Callable, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from itertools import product
import pandas as pd
import altair as alt
from itertools import product
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, RocCurveDisplay, roc_curve

interactions = load(Interactions, "outputs/interactions.json")()
def get_subcategories(category: DrugCategory):
    return [OMOPDrugs(sc) for sc in category.subcategories]

## Tools to compute predicted labels

# Policies to replace labels with no effect
def noop(_: DrugCategoryConditionInteractions) -> bool:
    return False

def exclude_weak(drug_condition_output: DrugCategoryConditionInteractions) -> bool:
    return drug_condition_output.evidence in ["none", "weak"]

def exclude_rare(drug_condition_output: DrugCategoryConditionInteractions) -> bool:
    return drug_condition_output.frequency  in ["none", "rare"]

def exclude_rare_or_weak(drug_condition_output: DrugCategoryConditionInteractions) -> bool:
    return exclude_rare(drug_condition_output) or exclude_weak(drug_condition_output)

def exclude_rare_and_weak(drug_condition_output: DrugCategoryConditionInteractions) -> bool:
    return exclude_rare(drug_condition_output) and exclude_weak(drug_condition_output)

def exclude_prob_or_rare_and_weak(drug_condition_output: DrugCategoryConditionInteractions) -> bool:
    return drug_condition_output.probability  in [0.01,0.1] or exclude_rare_and_weak(drug_condition_output)

def merge_subcategory_labels(labels: Iterable[OMOPLabels]) -> OMOPLabels:
    """Return the strongest label in the iterable; if two labels have opposite valence, return that there is no effect."""
    output = next(labels)
    for label in labels:
        if label != output and label != OMOPLabels.NO_EFFECT:
            if output == OMOPLabels.NO_EFFECT:
                output = label
            else:
                return OMOPLabels.NO_EFFECT
    return output

def predicted_omop_table(
    interactions: Interactions = interactions, 
    exclude_fn: Callable[[DrugCategoryConditionInteractions], bool] = noop,
    merge_fn: Callable[[Iterable[OMOPLabels]], OMOPLabels] = merge_subcategory_labels,
) -> dict[OMOPDrugs, dict[OMOPConditions, OMOPLabels]]:
    predicted = defaultdict[OMOPDrugs, dict[OMOPConditions, OMOPLabels]](dict)
    for drug in omop_drugs:
        drug_interactions = interactions.categories[str(drug.value)]
        for condition in conditions:
            drug_condition_output = drug_interactions.conditions[condition.value]
            if exclude_fn(drug_condition_output):
                label = OMOPLabels.NO_EFFECT
            else:
                label = OMOPLabels(drug_condition_output.net_effect)
            predicted[drug][condition] = label

    # use output OMOP drugs
    outputs = {}
    for drug in omop_drugs_for_evaluation:
        if isinstance(drug.value, DrugCategory):
            outputs[drug] = {
                condition: merge_fn(predicted[d][condition] for d in get_subcategories(drug.value))
                for condition in conditions
            }
        else:
            outputs[drug] = predicted[drug]
            
    return outputs

## Utilities to compute confidence scores

# Policies to compute confidence scores
# Confidence methods compute a score for the "increase" class and for the "effect" classes (when the target is "No Effect") 
def confidence(
    drug_condition_output: DrugCategoryConditionInteractions,
    target_class: Literal[OMOPLabels.NO_EFFECT, OMOPLabels.INCREASE]=OMOPLabels.NO_EFFECT,
) -> float:
    """
    Use provided confidence score (flip if no effect for "no effect" target, and negate if
    decrease when the target label is "Increase").
    """
    label = OMOPLabels(drug_condition_output.net_effect)
    if target_class == OMOPLabels.INCREASE:
        if label == OMOPLabels.NO_EFFECT:
            return 1 - drug_condition_output.confidence
        elif label == OMOPLabels.DECREASE:
            return - drug_condition_output.confidence
    elif target_class == OMOPLabels.NO_EFFECT and label == OMOPLabels.NO_EFFECT:
        return 1 - drug_condition_output.confidence

    return drug_condition_output.confidence

def probability(
    drug_condition_output: DrugCategoryConditionInteractions,
    target_class: Literal[OMOPLabels.NO_EFFECT, OMOPLabels.INCREASE]=OMOPLabels.NO_EFFECT,
) -> float:
    """
    Use provided probability score (which expresses the probability of
    any effect), negate if decrease and the target label is
    "Increase".
    """
    label = OMOPLabels(drug_condition_output.net_effect)
    if label == OMOPLabels.DECREASE and target_class == OMOPLabels.INCREASE: 
        return - drug_condition_output.probability
    return drug_condition_output.probability

def frequency_evidence(
    drug_condition_output: DrugCategoryConditionInteractions,
) -> float:
    """Order in lexicographic order by label, frequency, and evidence."""
    base_label = OMOPLabels(drug_condition_output.net_effect)
    evidence = drug_condition_output.evidence
    frequency = drug_condition_output.frequency

    evidence_levels = {"none": 0, "weak": 1, "strong": 2}
    frequency_levels = {"none": 0, "rare": 1, "common": 2}
    label_levels = {OMOPLabels.NO_EFFECT: 0, OMOPLabels.INCREASE: 1, OMOPLabels.DECREASE: 1}

    if base_label == OMOPLabels.NO_EFFECT:
        # As evidence increases, we are less confident in any effect
        score = sum(a * b for a, b in zip(
            [label_levels[base_label], frequency_levels[frequency], 2-evidence_levels[evidence]],
            [9, 3, 1]
        ))
    else:
        score = sum(a * b for a, b in zip(
            [label_levels[base_label], frequency_levels[frequency], evidence_levels[evidence]],
            [9, 3, 1]
        ))

    return score

def evidence_frequency(
    drug_condition_output: DrugCategoryConditionInteractions,
) -> float:
    """Order in lexicographic order by label, frequency, and evidence."""
    base_label = OMOPLabels(drug_condition_output.net_effect)
    evidence = drug_condition_output.evidence
    frequency = drug_condition_output.frequency

    evidence_levels = {"none": 0, "weak": 1, "strong": 2}
    frequency_levels = {"none": 0, "rare": 1, "common": 2}
    label_levels = {OMOPLabels.NO_EFFECT: 0, OMOPLabels.INCREASE: 1, OMOPLabels.DECREASE: 1}

    if base_label == OMOPLabels.NO_EFFECT:
        # As evidence increases, we are less confident in any effect
        score = sum(a * b for a, b in zip(
            [label_levels[base_label], 2-evidence_levels[evidence], frequency_levels[frequency]],
            [9, 3, 1]
        ))
    else:
        score = sum(a * b for a, b in zip(
            [label_levels[base_label], evidence_levels[evidence], frequency_levels[frequency]],
            [9, 3, 1]
        ))

    return score

def omop_table_effect_confidences(
    interactions: Interactions = interactions, 
    confidence_fn: Callable[[DrugCategoryConditionInteractions], float] = confidence,
    merge_fn: Callable[[Iterable[float]], float] = max,
) -> dict[OMOPDrugs, dict[OMOPConditions, float]]:
    predicted = defaultdict[OMOPDrugs, dict[OMOPConditions, float]](dict)
    for drug in omop_drugs:
        drug_interactions = interactions.categories[str(drug.value)]
        for condition in conditions:
            drug_condition_output = drug_interactions.conditions[condition.value]
            score = confidence_fn(drug_condition_output)
            predicted[drug][condition] = score
            
    # use output OMOP drugs
    outputs = {}
    for drug in omop_drugs_for_evaluation:
        if isinstance(drug.value, DrugCategory):
            outputs[drug] = {
                condition: merge_fn(predicted[d][condition] for d in get_subcategories(drug.value))
                for condition in conditions
            }
        else:
            outputs[drug] = predicted[drug]
            
    return outputs

## Tools for evaluation
def to_verified_labels(
    predicted: dict[OMOPDrugs, dict[OMOPConditions, OMOPLabels]],
    ground_truth: dict[OMOPDrugs, dict[OMOPConditions, OMOPLabels]] = omop_table, 
) -> tuple[list[OMOPLabels], list[OMOPLabels]]:
    ground_truth_labels = []
    predicted_labels = []

    for drug in omop_drugs_for_evaluation:            
        for condition in conditions:
            ground_truth_label = ground_truth[drug][condition]

            if ground_truth_label == OMOPLabels.NO_EFFECT:
                continue

            if ground_truth_label == OMOPLabels.NO_EFFECT_VERIFIED:
                ground_truth_label = OMOPLabels.NO_EFFECT
                
            predicted_label = predicted[drug][condition]

            ground_truth_labels.append(ground_truth_label)
            predicted_labels.append(predicted_label)

    return ground_truth_labels, predicted_labels

def to_true_pos_neg_with_predicted_confidence(
    predicted: dict[OMOPDrugs, dict[OMOPConditions, float]],
    ground_truth: dict[OMOPDrugs, dict[OMOPConditions, OMOPLabels]] = omop_table, 
    target_class: OMOPLabels = OMOPLabels.NO_EFFECT,
) -> tuple[list[bool], list[float]]:
    """
    Processes the samples into true positive (i.e. increase or decrease)
    and true negative (i.e. no-effect) with confidence scores when target
    class is "No Effect", else treats "Increase" only as true.
    """
    ground_truth_is_positive = []
    confidences = []

    for drug in omop_drugs_for_evaluation:            
        for condition in conditions:
            ground_truth_label = ground_truth[drug][condition]

            if ground_truth_label == OMOPLabels.NO_EFFECT:
                continue

            if ground_truth_label == OMOPLabels.NO_EFFECT_VERIFIED:
                ground_truth_label = OMOPLabels.NO_EFFECT

            if target_class == OMOPLabels.NO_EFFECT:
                is_positive = ground_truth_label != target_class
            else:
                is_positive = ground_truth_label == target_class

            predicted_confidence = predicted[drug][condition]

            ground_truth_is_positive.append(is_positive)
            confidences.append(predicted_confidence)

    return (ground_truth_is_positive, confidences)

def omop_f1(
    exclude_fn: Callable[[DrugCategoryConditionInteractions], bool] = noop,
) -> dict[str, float]:
    """Computes the F1 score for both effect and ADE transformations."""
    predictions = predicted_omop_table(interactions, exclude_fn)
    y, yhat = to_verified_labels(predictions)

    omop_f1s = {}

    def get_f1(label_map: dict[OMOPLabels, int]) -> float:
        y_bool = [label_map[yi] for yi in y]
        yhat_bool = [label_map[yi] for yi in yhat]
        return f1_score(y_bool, yhat_bool, average="binary")

    # Effect f1: treat Increase and Decrease as positive
    label_map = {
        OMOPLabels.DECREASE: 1,
        OMOPLabels.NO_EFFECT: 0,
        OMOPLabels.INCREASE: 1,
    }

    omop_f1s["Effect"] = get_f1(label_map)

    # ADE f1: treat Increase as positive
    label_map = {
        OMOPLabels.DECREASE: 0,
        OMOPLabels.NO_EFFECT: 0,
        OMOPLabels.INCREASE: 1,
    }
    omop_f1s["ADE"] = get_f1(label_map)

    return omop_f1s

    
# Modified from sklearn.metrics.ConfusionMatrixDisplay
def plot_confusion_matrix(
    cm, 
    ax=None, 
    cmap="viridis", 
    im_kw=None, 
    text_kw=None, 
    colorbar=True, 
    display_labels=None,
    xticks_rotation="horizontal",
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    n_classes = cm.shape[0]
    count = cm.sum(axis=1)
    cm_normalized = cm / count[:, None]
    
    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_kw = im_kw or {}
    im_kw = {**default_im_kw, **im_kw}
    text_kw = text_kw or {}
    
    im_ = ax.imshow(cm_normalized, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)
    
    text = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm_normalized.max() + cm_normalized.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm_normalized[i, j] < thresh else cmap_min
        text_cm = f"{cm_normalized[i,j]:.2g}\n({cm[i,j]:d}/{count[i]:d})"
        default_text_kwargs = dict(ha="center", va="center", color=color)
        text_kwargs = {**default_text_kwargs, **text_kw}

        text[i, j] = ax.text(j, i, text_cm, **text_kwargs)
    
    if display_labels is None:
        display_labels = np.arange(n_classes)
        
    if colorbar:
        fig.colorbar(im_, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    return fig, im_

def omop_confusion(exclude_fn, **kwargs):
    predictions = predicted_omop_table(interactions, exclude_fn)
    y, yhat = to_verified_labels(predictions)
    label_map = {
        OMOPLabels.DECREASE: 0,
        OMOPLabels.NO_EFFECT: 1,
        OMOPLabels.INCREASE: 2,
    }
    y = [label_map[yi] for yi in y]
    yhat = [label_map[yi] for yi in yhat]
    return plot_confusion_matrix(
        confusion_matrix(y, yhat), 
        display_labels=["Decrease", "No Effect", "Increase"],
        **kwargs
    )

def plot_sensitivity_specificity(confidence_method=confidence, show=True, path=None):
    _, ax = plt.subplots(1, 2, figsize=(10,4.5))
    
    for i, target_class in enumerate([OMOPLabels.NO_EFFECT, OMOPLabels.INCREASE]):
        predictions = omop_table_effect_confidences(interactions, partial(confidence_method, target_class=target_class))
        y, scores = to_true_pos_neg_with_predicted_confidence(predictions, target_class=target_class)
        y = np.array(y).astype(np.bool_)
        scores = np.array(scores)
        
        _, _, thresholds = roc_curve(y, scores)
        
        def sensitivity_specificity(threshold):
            """Sensitivity/specificity where the threshold for positive classification is set to `threshold`."""
            classify_positive = scores >= threshold
            classify_negative = ~classify_positive
        
            sensitivity = classify_positive[y].astype(np.float_).mean()
            specificity = classify_negative[~y].astype(np.float_).mean()
        
            return sensitivity, specificity
        
        sensitivity, specificity = zip(*[sensitivity_specificity(t) for t in thresholds])
    
        ax[i].plot(sensitivity, specificity)
        ax[i].set_xlabel("Sensitivity")
        ax[i].set_ylabel("Specificity")
        ax[i].set_title("Effect Sensitivity-Specificity Curve" if target_class == OMOPLabels.NO_EFFECT else "ADE Sensitivity-Specificity Curve")
    
    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()

def plot_rocs(confidence_method=confidence, show=True, path=None):
    _, ax = plt.subplots(1, 2, figsize=(10,4.5))
    for i, target_class in enumerate([OMOPLabels.NO_EFFECT, OMOPLabels.INCREASE]):
        predictions = omop_table_effect_confidences(interactions, partial(confidence_method, target_class=target_class))
        y, scores = to_true_pos_neg_with_predicted_confidence(predictions, target_class=target_class)
        fpr, tpr, _ = roc_curve(y, scores)
        plotter = RocCurveDisplay(fpr=fpr, tpr=tpr)
        plotter.plot(ax=ax[i])
        
        ax[i].set_title("Effect ROC Curve" if target_class == OMOPLabels.NO_EFFECT else "ADE ROC Curve")

    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()

def omop_auc(
        confidence_fn,
        target_class: Literal[OMOPLabels.NO_EFFECT, OMOPLabels.INCREASE]=OMOPLabels.NO_EFFECT,
):
    predictions = omop_table_effect_confidences(interactions, partial(confidence_fn, target_class=target_class))
    y, scores = to_true_pos_neg_with_predicted_confidence(predictions, target_class=target_class)
    return roc_auc_score(y, scores)


def plot_interactions(
        interactions: dict[OMOPDrugs, dict[OMOPConditions, OMOPLabels]],
        save_to: Optional[str]=None,
        width=800,
        domain = ["Decreased Risk", "No Effect", "Increased Risk", "No Effect (Evaluated)"],
        range_ = ["green", "white", "red", "blue"],
        keep_blue=True
) -> alt.Chart:
    records = []
    ground_truth = False
    for drug_category, category_interactions in interactions.items():
        for category, label in category_interactions.items():
            if label == OMOPLabels.NO_EFFECT_VERIFIED:
                ground_truth = True
                
            records.append({
                "Drug Category": printing_name(drug_category),
                "Condition": printing_name(category),
                "Label": printing_name(label) if label != OMOPLabels.NO_EFFECT_VERIFIED or keep_blue else printing_name(OMOPLabels.NO_EFFECT),
            })
    
    df = pd.DataFrame.from_records(records)
    
    chart = alt.Chart(df).mark_rect(stroke="black").encode(
        alt.X("Drug Category:N").sort(alt.SortArray([printing_name(d) for d in omop_drugs_for_evaluation])),
        alt.Y("Condition:N", title="Outcome").sort(alt.SortArray([printing_name(c) for c in conditions])),
        alt.Color("Label:N", scale=alt.Scale(domain=domain if ground_truth and keep_blue else domain[:-1], range=range_)),
    ).properties(
        width=width,
        height=200,
    )

    if save_to:
        chart = chart.configure_axisX(
            orient="top",
            labelAngle=-30,
        )
        chart.save(save_to)

    return chart

def predictions_summary(exclude_fn):
    for f1_type, f1_score in omop_score(exclude_fn):
        print(f"{f1_type} F1: {f1_score}")
        
    return omop_confusion(exclude_fn)
