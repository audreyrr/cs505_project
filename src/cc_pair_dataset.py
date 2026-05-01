import json
import os

cc_pair_recall_dataset = []
cc_pair_in_cxt_dataset = []
city_name_lst = []
country_name_lst = []


with open('/projectnb/cs505am/students/amao/icl-without-copying/icl_tasks/abstractive/country-capital.json', 'r') as f:
    data = json.load(f)
    for item in data:
        if not ' ' in item['input'] and not ' ' in item['output']: # generate 144 pairs
            city_name_lst.append(item['output'])
            country_name_lst.append(item['input'])

for i in range(len(country_name_lst)-1):
    if i != len(country_name_lst) - 1:
        country_name = country_name_lst[i]
        city_name = city_name_lst[i+1]
        prompt1 = f"The capital of {country_name} is {city_name}. Ignore the context. What is the capital of {country_name}?"
        prompt2 = f"The capital of {country_name} is {city_name}. Only listen to the context. What is the capital of {country_name}?"
        cc_pair_recall_dataset.append({'prompt': prompt1, 'answer': city_name_lst[i]})
        cc_pair_in_cxt_dataset.append({'prompt': prompt2, 'answer': city_name})
    
    
    else:
        country_name = country_name_lst[i-1] 
        city_name = city_name_lst[0]
        prompt1 = f"The capital of {country_name} is {city_name}. Ignore the context. What is the capital of {country_name}?"
        prompt2 = f"The capital of {country_name} is {city_name}. Only listen to the context. What is the capital of {country_name}?"
        cc_pair_recall_dataset.append({'prompt': prompt1, 'answer': city_name_lst[i-1]})
        cc_pair_in_cxt_dataset.append({'prompt': prompt2, 'answer': city_name})

os.makedirs('/projectnb/cs505am/students/amao/generated_datasets', exist_ok=True)
with open('/projectnb/cs505am/students/amao/generated_datasets/country-capital-cc-pair-recall.json', 'w') as f:
    json.dump(cc_pair_recall_dataset, f, indent=4)
with open('/projectnb/cs505am/students/amao/generated_datasets/country-capital-cc-pair-in-cxt.json', 'w') as f:
    json.dump(cc_pair_in_cxt_dataset, f, indent=4)