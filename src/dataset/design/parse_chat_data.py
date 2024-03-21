import json
import glob
import argparse


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
    