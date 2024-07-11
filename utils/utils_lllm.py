import re

def extract_task_name_from_text(text):
    # Assume the text is of the form "task_name (task description)"
    match = re.match(r"([^ ]+) \(", text)
    if match:
        return match.group(1)
    return None

def extract_list_from_text(text):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text)

    # Extract the list
    if match:
        return match.group(0)
    return None

