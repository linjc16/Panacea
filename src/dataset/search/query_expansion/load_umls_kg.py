from owlready2 import *
from tqdm import tqdm
import pdb


def pre_select_entities():
    with open('data/downstream/search/temp/concepts.txt', 'r') as f:
        entities = f.readlines()
    
    entities = set([x.strip() for x in entities])


    database_filepath = '/data/linjc/kgllm/pym.sqlite3'

    default_world.set_backend(filename=database_filepath)
    PYM = get_ontology("http://PYM/").load()
    CUI = PYM["CUI"]
    
    synonyms = {}
    for entity in tqdm(entities):
        try:
            entity = entity.strip()
            entity = CUI[entity]
            if entity.label:
                entity_label = str(entity.label[0])
                # check the number of words in the label
                if len(entity_label.split()) < 5 and len(entity.synonyms) > 1:
                    synonyms[entity] = [str(item) for item in entity.synonyms]
                    pdb.set_trace()
        except:
            continue

if __name__ == '__main__':
    pre_select_entities()
    pdb.set_trace()