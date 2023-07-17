import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

import random
from tqdm import tqdm
generated_count = 0
STEM = {
    "S": ["Category:Applied_sciences", "Category:Formal_sciences", "Category:Natural_sciences"],
    "T": ["Category:Technology"],
    "E": ["Category:Engineering_disciplines"],
    "M": ["Category:Fields_of_mathematics", "Category:Subfields_of_physics"]
}
import random
def split_category_members(members):
    category_list, page_list= [], []

    for member_name, member_page in members:
        if member_name.startswith('Category'):
            category_list.append((member_name, member_page))
        else:
            page_list.append((member_name, member_page))
    
    return category_list, page_list

def get_wiki_random_page(deep_subcategories=True):
    prob = random.random()
    stem_label, stem_categories = random.choice(list(STEM.items()))
    if prob > 0.6:
        stem_label, stem_categories = "S", STEM["S"]
    elif prob > 0.7:
        stem_label, stem_categories = "T", STEM["T"]
    elif prob > 0.8:
        stem_label, stem_categories = "E", STEM["E"]
    elif prob > 0.9:
        stem_label, stem_categories = "M", STEM["M"]
    category = random.choice(stem_categories)
    category_page = wiki_wiki.page(category)
    for _ in range(0,5):
        chosen_list = list(category_page.categorymembers.items())
        if deep_subcategories:
            category_list, page_list = split_category_members(chosen_list)
            chosen_list = []
        else:
            category_list, page_list = [], []

        # 50% change to select category or page list if one of them isn't empty
        # helps to go deeper into subcategories because there're more pages than categories
        if not (category_list or page_list) and not chosen_list:
            continue
        elif not category_list:
            chosen_list = page_list
        elif not page_list:
            chosen_list = category_list
        else:
            chosen_list = random.choice([category_list, page_list])

        # select random page from chosen list
        selected_page_name, selected_page = random.choice(chosen_list)

        if not selected_page_name.startswith("Category"):
            break
        
        category_page = selected_page
    if not selected_page_name.startswith("Category"):
        return None, None, None, None
    return selected_page, stem_label, category, category_page

import os
import json
def get_wiki_text():
    wiki_page, stem_label, category, category_page = get_wiki_random_page()
    if wiki_page is None:
        return None
    os.makedirs(f"/datadrive1/wiki/articles/{stem_label}", exist_ok=True)
    path = f"/datadrive1/wiki/articles/{stem_label}/{wiki_page.pageid}.json"
    
    if os.path.exists(path):
        return "Exists"
    
    data = {}
    data['pageid'] = wiki_page.pageid
    data['text'] = wiki_page.text
    data['title'] = wiki_page.title
    data['stem_label'] = stem_label
    data['category'] = category
    #data['category_page'] = category_page
    
    with open(path,"w") as f:
        json.dump(data, f)
    
    return "Done"
get_wiki_text()

pages_count = 2
max_completion_attempts = 10
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')

import random
from tqdm import tqdm
generated_count = 0
for i in tqdm(range(0,40_000), total=40_000):
    try:
        get_wiki_text()
    except:
        pass