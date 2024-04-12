# MALADE
This is a repository for the paper, MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance

## Set up environment and install dependencies
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

## Set up environment variables (API keys, etc)

To use the example scripts with an OpenAI LLM, you need an OpenAI API Key.
In the root of the repo, copy the `.env-template` file to a new file `.env`:
```bash
cp .env-template .env
```
Then insert any necessary key values in the '.env' file.

To test that everything is setup properly, run
```
```

## Run experiments

