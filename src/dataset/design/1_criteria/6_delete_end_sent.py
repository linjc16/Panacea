import json
import pdb

if __name__ == '__main__':
    filename = 'data/downstream/design/parsed/criteria/test.json'
    with open(filename, 'r') as f:
        data = json.load(f)

    dels = []

    for key, value in data.items():
        # index the even indices
        responses_end = value[-1]
        if responses_end['role'] == 'assistant':
            if "welcome" in responses_end['content'] or "good luck" in responses_end['content']:
                dels.append(
                    {key: True}
                )
                continue

        dels.append(
            {key: False}
        )

    
    # save to data/downstream/design/parsed/criteria/in_en_split_turn.json
    with open('data/downstream/design/parsed/criteria/del_end_sent.json', 'w') as f:
        json.dump(dels, f, indent=4)