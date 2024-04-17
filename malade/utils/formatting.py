from typing import Iterator

def format_list(xs: Iterator[str], numbered: bool=False, zero_base: bool=False) -> str:
    """
    Formats a list of strings into a numbered list if `numbered` else,
    as a comma separated list (in a natural language format, including
    "and").
    """
    if numbered:
        items = [
            f"{(i if zero_base else i+1)}. {x}"
            for i, x in enumerate(xs)
        ]
        newline = "\n" # Python < 3.12 f-string limitation
        return newline.join(items)
    
    xs_list = list(map(lambda s: s.strip(), xs))
    if len(xs_list) == 0:
        return ""
    elif len(xs_list) == 1:
        return xs_list[0]

    formatted = ", ".join(xs_list[:-1])
    formatted += f"{',' if len(xs_list) > 2 else ''} and {xs_list[-1]}"

    return formatted
