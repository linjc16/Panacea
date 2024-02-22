import sys
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
sys.path.append('./')

from src.utils.utils_search import build_search_expression, fetch_trials

# Define a function to process each search expression
def process_search_expression(exp):
    exp = exp.strip()
    try:
        trials = fetch_trials(exp)
    except:
        return None  # Return None if an exception occurs
    if len(trials) > 0:
        # Append the search expression to the file
        with open('data/downstream/search/query_search_exp.txt', 'a') as f:
            f.write(exp + '\n')
    return exp  # Return the expression for progress tracking

def pre_select():
    with open('data/downstream/search/temp/query_search_exp.txt', 'r') as f:
        search_expression = f.readlines()

    # Determine the number of processes to use
    num_processes = 4
    
    # Use a Pool to parallelize the operationc
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_search_expression, search_expression), total=len(search_expression)):
            pass  # tqdm will automatically update the progress

def deduplicate():
    with open('data/downstream/search/query_generation/query_search_exp.txt', 'r') as f:
        search_expression = f.readlines()
    
    search_expression = list(set(search_expression))
    with open('data/downstream/search/query_generation/query_search_exp.txt', 'w') as f:
        f.writelines(search_expression)

if __name__ == '__main__':
    # pre_select()
    deduplicate()
