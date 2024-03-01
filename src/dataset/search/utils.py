import sys
import os

sys.path.append('./')
from src.utils.utils_search import build_search_expression

import pdb



if __name__ == '__main__':
    json_query = """
    {
        "phase": "Phase1",
        "study_type": "Interventional",
        "diseases": ["Diabetes", "Hypertension"],
        "start_year": {"YEAR": "2022", "OPERATOR": "on"}
    }
    """
    import json
    query_data = json.loads(json_query)
    
    search_expression = build_search_expression(query_data)

    pdb.set_trace()