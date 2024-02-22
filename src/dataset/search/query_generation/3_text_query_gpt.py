import os
import json
import sys
sys.path.append('./')

from src.utils.gpt_azure import gpt_chat
from tqdm import tqdm

INPUT_PROMPT = (
    "You are asked to generated a text-formatted query given a search expression used for searching clinical trials in a database.\n"
    "The search expression is a string that contains one or more of the following fields: Condition, InterventionType, LeadSponsorName, OverallStatus, Phase, StudyType, ResponsiblePartyInvestigatorFullName, NCTId, LocationCountry, StartDate, and CompletionDate. "
    "The search expression may contain multiple fields and each field may contain multiple values.\n"
    "The generated query should contain the information from the search expression in a human-readable format. "
    "The query can be converted back to the search expression. The generated query should imitate user's natural language and be as informative as possible. "
    "The generated query style should be diverse.\n\n"
    # "Example:\n"
    # "Search expression: AREA[Condition]\"endometriosis\" AND AREA[OverallStatus]RECRUITING AND AREA[StudyType]OBSERVATIONAL;"
    # "Generated query: I require a list of currently recruiting observational studies focused on the condition of endometriosis. Could you assist?\n"
    # "Search expression: AREA[Condition]\"hypertension\" OR AREA[Condition]\"high blood pressure\");"
    # "Generated query: Researching studies focused on managing hypertension or high blood pressure.\n\n"
    # "Below is the search expression you need to generate a query for.\n"
    "Search expression: {search_expression}\n"
    "Generated query:"
)

def query_generation(search_expression: str) -> str:
    # Generate the query
    query = gpt_chat(prompt=INPUT_PROMPT, query_dict={'search_expression': f"{search_expression}"})
    return query

def main():
    with open('data/downstream/search/query_generation/query_search_exp_pairs.json', 'r') as f:
        data_dict = json.load(f)
    
    save_file = 'data/downstream/search/query_generation/output.json'
    i = 0
    for key, value in tqdm(data_dict.items()):
        search_exp = value['search_exp']
        try:
            output = query_generation(search_exp)
        except:
            output = ""
        data_dict[key]['query'] = output
        if i % 100 == 0:
            with open(save_file, 'w') as f:
                json.dump(data_dict, f, indent=4)
        i += 1
    
    with open(save_file, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
        
if __name__ == "__main__":
    main()