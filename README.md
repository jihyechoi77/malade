# **MALADE: <u>M</u>ultiple <u>A</u>gents powered by <u>L</u>LMs for <u>ADE</u> Extraction (MLHC'24)**

[![](https://img.shields.io/badge/Paper-pink?style=plastic&logo=GitBook)](https://arxiv.org/abs/2408.01869v1)
[![](https://img.shields.io/badge/Blog-pink?style=plastic&logo=twitter)](https://langroid.github.io/langroid/blog/2024/08/12/malade-multi-agent-architecture-for-pharmacovigilance/)
[![](https://img.shields.io/badge/Twitter-pink?style=plastic&logo=twitter)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for the paper:

Jihye Choi, Nils Palumbo, Prasad Chalasani, Matthew Engelhard, Somesh Jha, Anivarya Kumar, & David Page, (2024). 
*MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance.* 
[Machine Learning for Healthcare 2024](https://www.mlforhc.org/).


## **💊 What is MALADE?**

MALADE (pronounced like the French word [<em>malade</em>](http://www.french-linguistics.co.uk/how-to-pronounce/malade-ed6e4e308efdca82/) meaning 'sick' or 'ill') is a framework for the orchestration of Large Language Model (LLM)-powered agents with Retrieval Augmented Generation (RAG)
for Pharmacovigilance, in particular for Adverse Drug Event (ADE) extraction.

The core function of MALADE is to answer category-outcome ADE questions of the form:
> Does drug category X cause adverse event Y?,

or

> Is drug category X associated with adverse event Y?

For example, "Do ACE inhibitors cause angioedema?".

<p align="center">
    <br>
    <img src="./img/malade.png" width="1000"/>
    <br>    
<p>


The primary data source used is FDA Drug Label data as obtained via the 
OpenFDA API. Optionally, one can use the MIMIC-IV EHR data to 
identify the most representative drugs within a category (this is important
since FDA label data is specific to individual drugs, not categories).

For a given drug-category and outcome, MALADE produces a variety of qualitative and quantitative outputs, for example:
> **Label:** ACE inhibitors *increase* angioedema risk,\
> **Confidence:** 0.9 (i.e. confidence in the label),\
> **Frequency:** rare, \
> **Evidence:** strong, \
> **Justification**: The evidence from FDAHandler and drug labels for LISINOPRIL, CAPTOPRIL, and ENALAPRIL MALEATE consistently reports an increased risk of angioedema with the use of these ACE inhibitors. The incidence of angioedema is reported as rare, with occurrences such as one in 1000 patients for CAPTOPRIL. The evidence is considered strong due to the authoritative nature of the sources.

MALADE is evaluated against the OMOP Common Data Model (CDM) 
[ground truth table](https://www.niss.org/sites/default/files/Session3-DaveMadigan_PatrickRyanTalk_mar2015.pdf), which shows established category-outcome associations
for a specific set of 10 drug categories and 10 outcomes.

## **⚙️ Set up environment and install dependencies**
We leverage the awesome [Langroid](https://github.com/langroid/langroid) 
open-source Python library for multi-agent LLM applications.

IMPORTANT: Please ensure you are using Python 3.11+. If you are using poetry,
you may be able to just run `poetry env use 3.11` if you have Python 3.11 available in your system.

```bash
# clone this repository 
git clone [this-repository]
cd malade
```

Environment setup with conda:
```bash
# create empty environment:
conda env create -n malade python=3.11 -c conda-forge
conda activate malade
```

Setup with venv:
```bash
# create a virtual env under project root, .venv directory
python3 -m venv .venv

# activate the virtual env
. .venv/bin/activate
```

Install dependencies with Poetry:
```bash
# Optionally: poetry lock
poetry install
```

## **🔑 Set up environment variables (API keys, etc)**

To use the example scripts with an OpenAI LLM, you need an OpenAI API Key.
In the root of the repo, copy the `.env-template` file to a new file `.env`:
```bash
cp .env-template .env
```

First, an [OpenAI API key](https://platform.openai.com/docs/quickstart) is required;
save it in the `.env` file as `OPENAI_API_KEY=...` (no quotes).

A Qdrant instance and API key is required (see the [Langroid instructions](https://github.com/langroid/langroid?tab=readme-ov-file#set-up-environment-variables-api-keys-etc)); set up `QDRANT_API_URL` and `QDRANT_API_KEY` in `.env` as described there. 

An OpenFDA API key is also required (get one [here](https://open.fda.gov/apis/authentication/)), set it as 
`OPENFDA_API_KEY=...` in the `.env` file.

### **(Optional) Setup for drug representative generation**

This step is required only to run `DrugFinder` and the process to find
representative drugs in a category based on MIMIC-IV data.

Make sure that MIMIC-IV is installed and running on your machine as PostgreSQL database.
The MIMIC-IV can be obtained [here](https://physionet.org/content/mimiciv/2.2/#files). \
Access requires completing the following training described [here](https://physionet.org/content/mimiciv/view-required-training/2.2/#1). \
Instructions and code for loading MIMIC-IV into PostgreSQL are [here](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres). \
Finally, ensure that your user account has access to the `mimiciv` database.



## **🗂️ Code Structure**
We provide brief descriptions for each file as follows:

| Directory/File              | Description                                                                                                                                         |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `malade/`                   | core directory for codes                                                                                                                            |
| `malade/omop.py`             | define the OMOP Ground Truth table, and the associated drug categories and conditions                                                               |
| `malade/drug_categories.py`  | find representative drugs                                                                                                                           |
| `malade/omop_interactions.py` | contain `CategoryAgent` and `DrugAgent` <br/> identify drug-outcome associations and label drug category-outcome associations |
| `malade/critic_agent.py`     | contain `Critic` and `malade/omop_evaluation.py` <br/> contain utilities for evaluation (for use by `scripts/generate_results.py`)                  |
| `malade/doc/`                | contain RAG-related code                                                                                                                            |
| `malade/doc/fda_handler.py`  | contain `FDAHandler`                                                                                                                                |
| `malade/utils/`              | for general utilities                                                                                                                               |
| `malade/utils/openfda.py`     | for the OpenFDA query code                                                                                                                          |
| `malade/tools/`               | contain utilities related to tool-use                                                                                                               |

### Run Experiments

**TODO: add brief demo for each step below.**

* STEP1: Finding Representative Drugs (Optional)

If MIMIC-IV was set up, run `DrugFinder` and the drug category representative identification process with
```angular2html
python3 malade/drug_categories.py --recompute
```
<!---
> Mention what this does? i.e. it finds representative drugs for the category mentioned 
> by the user when they interact with this? And where is output shown/stored?
> Show example interaction/output (maybe a screenshot or a video?)
--->

* STEP2: Identifying Drug-Outcome Associations 

Run `DrugAgent` and the drug-outcome association identification process with
```angular2html
python3 malade/omop_interactions.py --recompute_interactions
```
<!---
> Clarify that user will be prompted to enter a drug-category and outcome, or can they pass these
> as cli arguments? And where is output shown/stored? Show example interaction/output 
> (screenshots or video)
--->

* STEP3: Labeling Drug Category-Outcome Associations
<!---
> What does it mean to "label" the category-outcome, i.e. how does this differ from 
> previous step, i.e. "identification". Does "label" mean a score is produced?
> Show example interaction (screenshots or video).
--->
Run `CategoryAgent` and the category-outcome labeling process with 
```angular2html
python3 malade/omop_interactions.py --recompute_labels
```
Run `python3 scripts/generate_summary_files.py` to process the outputs from MALADE into a readable format.\
`scripts/generate_results.py` contains the code to generate the final experimental results. 

## **🔍 Outputs of MALADE**

The outputs from MALADE are in the `outputs/` directory;

| File                    | Description                                                        |
|----------------------------------|--------------------------------------------------------------------|
| `outputs/representative_drugs.json` | outputs from `DrugFinder`                                          |
| `outputs/interactions.json`        | outputs from `DrugAgent` and `CategoryAgent` |
| `outputs/representative_drugs.md`  | outputs from `DrugAgent` in a readable format                      |
| `outputs/omop_results.md`          | outputs from `CategoryAgent` in a readable format       |

The logs generated by the agents are in the `logs/` directory; the path is of the form \
`logs/DrugFinder-{category name}.log` for `DrugFinder`, \
`logs/DrugOutcomeInfoAgent-{outcome}-{drug name}.log` for `DrugAgent`, and \
`logs/CategoryOutcomeRiskAgent-{outcome}-{category name}.log` for `CategoryAgent`.

## **📎 Reference**

If you find this code/work useful in your own research, please consider citing the following:
```bibtex
@misc{choi2024malade,
      title={MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance}, 
      author={Jihye Choi and Nils Palumbo and Prasad Chalasani and Matthew M. Engelhard and Somesh Jha and Anivarya Kumar and David Page},
      year={2024},
      eprint={2408.01869},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.01869}, 
}
```

