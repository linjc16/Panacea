import json



if __name__ == '__main__':
    with open('data/downstream/matching/patient2trial/cohort/test_cot.json', 'r') as f:
        inputs = json.load(f)

    

    inputs_new = {}
    for key, value in inputs.items():
        # change input "Let's think step by step. \nFinally, you should always repeat" -> "Finally, you only need to output"
        value['input'] = value['input'].replace("Let's think step by step. \nFinally, you should always repeat", "Finally, you need to output only")
        inputs_new[key] = value
    
    with open('data/downstream/matching/patient2trial/cohort/test.json', 'w') as f:
        json.dump(inputs_new, f, indent=4, sort_keys=True)