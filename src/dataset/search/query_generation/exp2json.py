import re
from typing import Dict, Any

def parse_search_expression(expression: str) -> Dict[str, Any]:
    parts = re.split(r' AND |\) AND \(', expression.strip("()"))
    
    json_result = {}
    for part in parts:
        if 'RANGE' in part:
            match = re.search(r'RANGE\[(.*?)\]\[(.*?), (.*?)\]', part)
            field_area, start_date, end_date = match.groups()
            year, operator = parse_year_range(field_area, start_date, end_date)
            json_result[get_json_key(field_area)] = {"YEAR": year, "OPERATOR": operator}
        elif 'AREA' in part:
            field_area, values = parse_area_expression(part)
            json_key = get_json_key(field_area)
            if json_key in json_result:
                if not isinstance(json_result[json_key], list):
                    json_result[json_key] = [json_result[json_key]]
                json_result[json_key].extend(values)
            else:
                json_result[json_key] = values if len(values) > 1 else values[0]
    
    return json_result

def parse_year_range(field_area: str, start_date: str, end_date: str):
    year, operator = None, None
    if start_date == 'MIN':
        year = int(end_date.split('/')[-1]) 
        operator = 'before'
    elif end_date == 'MAX':
        year = int(start_date.split('/')[-1]) - 1 
        operator = 'after'
    else:
        year = int(re.search(r'\d{4}', start_date).group())
        operator = 'on'
    return year, operator

def parse_area_expression(expression: str):
    field_area, values_str = re.match(r'AREA\[(.*?)\](.*)', expression).groups()
    values = re.findall(r'"(.*?)"', values_str)
    return field_area, values

def get_json_key(field_area: str) -> str:
    field_to_area = {
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
    return field_to_area.get(field_area, field_area)


search_expression = 'AREA[Condition]"Diabetes" AND AREA[Phase]"PHASE2"'
json_output = parse_search_expression(search_expression)
print(json_output)
