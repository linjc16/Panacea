import json

# Load the JSON file data
with open('data/analysis/icd10/icd10_matching_results.json', 'r') as file:
    data = json.load(file)

# select the top 100 conditions, the first 100 conditions
data = {k: v for k, v in list(data.items())[:100]}

# Define a function to process and organize the data
def process_data(data):
    hierarchy = {}
    for code, details in data.items():
        chapter = details['chapter']
        section = details['section']
        

        # Ensure section has the correct format, duplicate if needed
        if len(section) == 3:
            section = f"{section}-{section}"

        if chapter not in hierarchy:
            hierarchy[chapter] = {}

        if section not in hierarchy[chapter]:
            hierarchy[chapter][section] = []

        hierarchy[chapter][section].append(code)

    
    return hierarchy


# Converts the hierarchy to a Newick tree format
def hierarchy_to_newick(hierarchy):
    def newick_format(sub_hierarchy, level=0):
        result = ""
        for key in sorted(sub_hierarchy.keys()):
            codes = sub_hierarchy[key]
            if isinstance(codes, dict):
                result += f"({newick_format(codes, level+1)}){key.replace('-', '~')}"
            else:
                code_list = ','.join(codes)
                result += f"({code_list}){key.replace('-', '~')}"
            result += ','
        return result.rstrip(',')
    return f"({newick_format(hierarchy)})root;"

# Save Newick tree to a file
def save_newick(newick_tree, output_path):
    with open(output_path, 'w') as file:
        file.write(newick_tree)
    return output_path


# Process the data
processed_hierarchy = process_data(data)

# Save the processed data to a JSON file
with open('data/analysis/icd10/icd10_top100_tree.json', 'w') as file:
    json.dump(processed_hierarchy, file, indent=4)

newick_output_path = 'data/analysis/icd10/icd10_hierarchy_newick_tree.nwk'
newick_tree = hierarchy_to_newick(processed_hierarchy)
output_file_path = save_newick(newick_tree, newick_output_path)