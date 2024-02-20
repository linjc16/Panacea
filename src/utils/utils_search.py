from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum
from pytrials.client import ClinicalTrials
import pandas as pd
import pdb


ct = ClinicalTrials()

json_schema = {
    "type": "object",
    "properties": {
        "diseases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The disease, disorder, syndrome, illness, or injury that is being studied. On ClinicalTrials.gov, conditions may also include other health-related issues, such as lifespan, quality of life, and health risks.",
        },
        "interventions": {
            "enum": ["", "Drug", "Biological", "Device", "Procedure", "Behavioral", "Dietary Supplement", "Other"],
            "type": "array",
            "items": {"type": "string"},
            "description": "A process or action that is the focus of a clinical study. Interventions include drugs, medical devices, procedures, vaccines, and other products that are either investigational or already available. Interventions can also include noninvasive approaches, such as education or modifying diet and exercise.",
        },
        "sponsor": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Sponsor name"
        },
        "status": {
            "enum": ["", "RECRUITING", "TERMINATED", "APPROVED_FOR_MARKETING", "COMPLETED", "ENROLLING_BY_INVITATION"],
            "type": "string",
            "description": "Overall status of the study"
        },
        "phase": {
            "enum": ["", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
            "type": "string",
            "description": "Phase of the study"
        },
        "study_type": {
            "enum": ["", "INTERVENTIONAL", "OBSERVATIONAL"],
            "type": "string",
            "description": "Type of the study"
        },
        "person_name": {
            "type": "string",
            "description": "Name of the investigator"
        },
        "nctid": {
            "type": "array",
            "items": {"type": "string"},
            "description": "NCT ID of the clinical trial study"
        },
        "locations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Name of the country or city"
        },
        "start_year": {
            "type": "object",
            "properties": {
                "YEAR": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Year"
                },
                "OPERATOR": {
                    "type": "string",
                    "enum": ["before", "after", "on"],
                    "description": "Enable detect before and after the year"
                }
            },
            "required": ["YEAR", "OPERATOR"],
            "description": "Start year of the trial"
        },
        "end_year": {
            "type": "object",
            "properties": {
                "YEAR": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Year"
                },
                "OPERATOR": {
                    "type": "string",
                    "enum": ["before", "after", "on"],
                    "description": "Enable detect before and after the year"
                }
            },
            "required": ["YEAR", "OPERATOR"],
            "description": "End year of the trial"
        }
    },
}


def build_search_expression(fields: Dict[str, Any]) -> str:
    # https://classic.clinicaltrials.gov/api/info/search_areas?fmt=XML 
    field_to_area = {
        "diseases": "Condition",
        "interventions": "InterventionType",
        "sponsor": "LeadSponsorName",
        "status": "OverallStatus",
        "phase": "Phase",
        "study_type": "StudyType",
        "person_name": "ResponsiblePartyInvestigatorFullName",
        "nctid": "NCTId",
        "locations": "LocationCountry",
        "start_year": "StartDate",
        "end_year": "CompletionDate",
    }

    expressions = []
    for key, value in fields.items():
        if key in ["start_year", "end_year"]:
            operator = value["OPERATOR"]
            year = int(value["YEAR"])
            if operator == "after":
                expressions.append(f'RANGE[{field_to_area[key]}][01/01/{year+1}, MAX]')
            elif operator == "before":
                expressions.append(f'RANGE[{field_to_area[key]}][MIN, 12/31/{year-1}]')
            elif operator == "on":
                expressions.append(f'AREA[{field_to_area[key]}]"{year}"')
        else:
            area = field_to_area.get(key, key) 
            if isinstance(value, list):
                or_expressions = " OR ".join([f'AREA[{area}]"{val}"' for val in value])
                expressions.append(f"({or_expressions})")
            else:
                expressions.append(f'AREA[{area}]{value}')
    
    search_expression = " AND ".join(expressions)
    return search_expression

def fetch_trials(search_expression: str) -> pd.DataFrame:
    trials = ct.get_study_fields(
        search_expr=search_expression,
        fields=["NCTId"],
        max_studies=500,
        fmt='csv'
    )
    
    df = pd.DataFrame.from_records(trials[1:], columns=trials[0])
    
    return df