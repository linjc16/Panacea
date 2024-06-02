import argparse
import json
import re


def process_errors(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    results = {}
    non_responsive_pattern = "Non-responsive Error"
    reason_type_pattern = r"Reason type: (\[[^\]]+\])"

    for key, text in data.items():
        if non_responsive_pattern in text:
            results[key] = -1
        else:
            match = re.search(reason_type_pattern, text)
            if match:
                # Convert the matched string to a list using eval
                results[key] = eval(match.group(1))
            else:
                results[key] = None  # In case no reason type is found

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2-7b')