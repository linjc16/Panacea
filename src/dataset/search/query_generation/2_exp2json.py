import re
from typing import Dict, Any, List
import pdb


# Function to parse each line and convert to dictionary format
def parse_line_to_dict(line: str) -> Dict[str, Any]:
    # Mapping from the text to the field names in JSON schema
    field_to_json_field = {
        "Condition": "diseases",
        "InterventionType": "interventions",
        "LeadSponsorName": "sponsor",
        "OverallStatus": "status",
        "Phase": "phase",
        "StudyType": "study_type",
        "ResponsiblePartyInvestigatorFullName": "person_name",
        "NCTId": "nctid",
        "LocationCountry": "locations",
        "StartDate": "start_year",
        "CompletionDate": "end_year",
    }
    
    # Initial empty dictionary to store the extracted information
    parsed_dict = {key: [] for key in field_to_json_field.values()}
    parsed_dict["start_year"] = {"YEAR": 0, "OPERATOR": ""}
    parsed_dict["end_year"] = {"YEAR": 0, "OPERATOR": ""}

    # Regular expression to extract information
    pattern = re.compile(r'(?:AREA\[(.*?)\]"?(.*?)"?)(?: OR | AND |$)|RANGE\[(.*?)\]\[(.*?), (.*?)\]')
    matches = pattern.findall(line)
    
    for match in matches:
        field, value, date_field, start_date, end_date = match
        if field:  # Regular fields
            json_field = field_to_json_field.get(field, field)
            if json_field in ["diseases", "interventions", "sponsor", "status", "phase", "study_type", "person_name", "nctid", "locations"]:
                parsed_dict[json_field].append(value.strip("\""))
        elif date_field:  # Date ranges
            json_field = field_to_json_field.get(date_field, date_field)
            if json_field in ["start_year", "end_year"]:
                year = int(end_date.split('/')[-1]) if end_date != "MAX" else int(start_date.split('/')[-1])
                operator = "before"
                if start_date == "MIN":
                    operator = "before"
                    year = int(end_date.split('/')[-1])
                elif end_date == "MAX":
                    operator = "after"
                else:  # Assuming "on" if specific year is provided without MIN or MAX
                    operator = "on"
                parsed_dict[json_field] = {"YEAR": year, "OPERATOR": operator}

    # Convert lists to single values if only one item present, except for interventions and status
    for key, value in parsed_dict.items():
        if not isinstance(value, dict) and len(value) == 1 and key not in ["diseases", "interventions", "status", "phase", "study_type"]:
            parsed_dict[key] = value[0]

    return parsed_dict

if __name__ == "__main__":
    filepath = 'data/downstream/search/query_generation/query_search_exp.txt'
    with open(filepath, 'r') as f:
        sample_lines = f.readlines()

    # Convert each line to dict and print
    parsed_dicts = [parse_line_to_dict(line) for line in sample_lines]

    saved_dicts = {}
    for i, (line, parsed_dict) in enumerate(zip(sample_lines, parsed_dicts)):
        # change the parsed_dict to string, also use "" not ''
        parsed_dict_str = str(parsed_dict)
        # use "" not ''
        parsed_dict_str = parsed_dict_str.replace("'", '"')
        
        saved_dicts[f"{i}"] = {"search_exp": line.strip(), "parsed_dict": parsed_dict_str}
         
    # save the pairs to a json file
    import json
    with open('data/downstream/search/query_generation/query_search_exp_pairs.json', 'w') as f:
        # convert the parsed_dict as string
        json.dump(saved_dicts, f, indent=4)