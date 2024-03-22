import json
import glob
import argparse
import re
import os
import pdb
from tqdm import tqdm

def format_dialogue(content):
    """
    Format the conversation content into dialogue pairs of User and Chatbot without relying on explicit line breaks.
    """

    user_role, assis_role = 'User', 'Chatbot'
    dialogue_pairs = []
    current_pair = {}
    prev_role = None

    # Split the content by role identifiers, keeping the delimiter
    parts = content.split(f'{user_role}:')
    for part in parts[1:]:
        sub_parts = part.split(f'{assis_role}:')
        user_text = sub_parts[0].strip()
        if user_text:
            if prev_role == user_role:
                dialogue_pairs.append(current_pair)
                current_pair = {}
            current_pair[user_role] = user_text
            prev_role = user_role

        for sub_part in sub_parts[1:]:
            assis_text, next_user_text = sub_part.rsplit(f'{user_role}:', 1) if f'{user_role}:' in sub_part else (sub_part, "")
            assis_text = assis_text.strip()
            if assis_text:
                current_pair[assis_role] = assis_text
                dialogue_pairs.append(current_pair)
                current_pair = {}
                prev_role = assis_role

            if next_user_text:
                current_pair[user_role] = next_user_text.strip()
                prev_role = user_role

    # Append the last pair if it exists
    if current_pair.get(user_role) or current_pair.get(assis_role):
        dialogue_pairs.append(current_pair)

    return dialogue_pairs

def convert_to_requested_format(dialogue_pairs):
    """
    Convert the dialogue pairs into the requested format with "content" and "role" keys.
    """
    formatted_output = []
    for pair in dialogue_pairs:
        if 'User' in pair:
            formatted_output.append({
                "content": pair['User'],
                "role": "user"
            })
        if 'Chatbot' in pair:
            formatted_output.append({
                "content": pair['Chatbot'],
                "role": "assistant"
            })
    return formatted_output


def check_valid(conversation):
    assert isinstance(conversation, list), "Each conversation must be a list"
    assert len(conversation) > 0, "Each conversation must have at least one message"
    for message in conversation:
        assert isinstance(message, dict), "Each message in a conversation must be a dictionary"
        assert 'content' in message and 'role' in message, "Each message must have 'content' and 'role' keys"
        assert isinstance(message['content'], str), "The 'content' key must be a string"
        assert message['role'] in ['user', 'assistant'], "The 'role' key must be either 'user' or 'assistant'"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='study_arms')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    file_dir = f'data/downstream/design/raw/reasons/{args.task}/{args.split}/chat/'

    filepaths = glob.glob(file_dir + '*.json')

    data_dict = {}
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            data_dict.update(json.load(f))
    
    parsed_dict = {}
    
    for key, value in tqdm(data_dict.items()):
        # if the start of the value is [, then load it as a list
        if not value:
            continue
        if value.startswith('['):
            try:
                parsed_dict[key] = json.loads(value.replace("\n", "\\n"))
            except:
                parsed_dict[key] = json.loads(value)
            finally:
                continue
        else:
            parsed_dict[key] = convert_to_requested_format(format_dialogue(value))

    
    parsed_dict_filtered = {}
    for key, value in parsed_dict.items():
        try:
            check_valid(value)
            parsed_dict_filtered[key] = value
        except:
            continue
    
    print(f"Number of keys in the parsed dictionary: {len(parsed_dict_filtered)}")
    print("Number of skipped keys: ", len(data_dict) - len(parsed_dict_filtered))

    # from datasets import Dataset

    # data_list = []
    # for key, value in tqdm(parsed_dict_filtered.items()):
    #     data_list.append(value)
    
    # data_dict = {'messages': data_list}
    # raw_datasets = Dataset.from_dict(data_dict, split='train')

    save_dir = f'data/downstream/design/parsed/{args.task}'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'{args.split}.json'), 'w') as f:
        json.dump(parsed_dict_filtered, f, indent=4)
