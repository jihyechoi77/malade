import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fire import Fire
import os
from typing import Optional
load_dotenv()


def get_drug_label_details(drug:str) -> list[dict]:
    api_key = os.getenv('OPENFDA_API_KEY')
    drug = drug.upper().replace(" ", "+")
    search_url = (
        f"https://api.fda.gov/drug/label.json?api_key={api_key}&"
        f'search=openfda.brand_name:"{drug}"+openfda.generic_name:"{drug}"'
    )
    response = requests.get(search_url)
    results = response.json().get("results", [])

    return results


def extract_html_text(html_content: str) -> Optional[str]:
    """
    Extracts and returns clean text from provided HTML content.

    Args:
    - html_content (str): A string containing HTML/XML content to be cleaned.

    Returns:
    - Optional[str]: A string with the extracted clean text, or
       None if no text is extracted.
    """
    # Initialize BeautifulSoup with lxml parser for better performance
    soup = BeautifulSoup(html_content, 'lxml')

    # Find all <paragraph> elements within <td> tags
    paragraphs = soup.find_all(
        'td',
        {'align': 'center',
         'styleCode': lambda value: 'Rrule' in value if value else False
         }
    )

    # Extract and concatenate the text from each paragraph
    clean_text = ' '.join(paragraph.get_text(strip=True) for paragraph in paragraphs)

    return clean_text if clean_text else None


def get_names(result: dict) -> set[str]:
    """Gets the brand and generic names from a single OpenFDA result."""
    results = set()

    for key in ["brand_name", "brand_name_base", "generic_name"]:
        if key in result:
            value = result[key]
            if isinstance(value, str):
                results.add(value.upper())

    return results


def get_drugs_of_class(drug_class: str, page_size: int=1000) -> set[str]:
    """
    Returns a list of all drug names (generic and brand names) belonging
    to a class.
    """
    api_key = os.getenv('OPENFDA_API_KEY')
    drug_classes = drug_class.upper().split(' ')

    def includes_all_terms_query(field: str) -> str:
        query = "+AND+".join(
            f'{field}:"{dc}"'
            for dc in drug_classes
        )
        return f"({query})"

    class_fields = [
        f"openfda.pharm_class_{class_type}"
        for class_type in ["cs", "epc", "pe", "moa"]
    ] + ["pharm_class", "generic_name", "brand_name", "brand_name_base"]

    search_url = (
        f"https://api.fda.gov/drug/ndc.json?api_key={api_key}&search="
        + "+".join(includes_all_terms_query(f) for f in class_fields)
        + f"&limit={page_size}"
    )
    names = set()
    done = False

    while not done:
        response = requests.get(search_url)
        response_json = response.json()

        if "results" in response_json:
            results = response_json["results"]
            headers = response.headers

            names = names.union(*(get_names(r) for r in results))

            # This is a search URL for the next page of results (if it exists)
            if "Link" in headers:
                link = headers["Link"]
                end = link.index(">")
                search_url = link[1:end]
            else:
                done = True
        else:
            done = True

    return names


def get_drugs_of_classes(*drug_classes: str) -> set[str]:
    """Gets all drugs in the provided classes."""
    return set().union(*(get_drugs_of_class(dc) for dc in drug_classes))

if __name__ == "__main__":
    Fire(get_drugs_of_classes)
