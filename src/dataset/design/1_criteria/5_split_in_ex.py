import json
import pdb

if __name__ == '__main__':
    filename = 'data/downstream/design/parsed/criteria/test.json'
    with open(filename, 'r') as f:
        data = json.load(f)

    boundary_turn = []
    for key, value in data.items():
        # index the even indices
        responses_ass = value[1::2]
        for idx, resp in enumerate(responses_ass):
            if idx < 3: 
                continue
            if 'exclusion criteria' in resp['content']:
                boundary_turn.append(
                    {key: idx}
                )
                break
    
    # save to data/downstream/design/parsed/criteria/in_en_split_turn.json
    with open('data/downstream/design/parsed/criteria/in_en_split_turn.json', 'w') as f:
        json.dump(boundary_turn, f, indent=4)