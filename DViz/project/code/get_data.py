import requests
import re
import yaml
import pandas as pd
import numpy as np

#------------------------------------------------------------------------------------
# get raw results 
# in: nyt house of representative election results
# out: two dataframes
#      one for district level election results info
#      anotehr for district level candidate results info
#
def get_results(source_url):

    r = requests.get(source_url)

    # fetch url for recipe collection and name of collection
    l = re.findall(r'eln_races =.*}],', r.text, re.S)
    s = l[0] 
    s = s[13:-2]
    s.replace('true','True')
    s.replace('false','False')
    guid_list = re.findall(r'{"guid":.*?"key_race":.*?}', r.text, re.S)

    # will convert list of strings into two lists of dictionaries
    # one for district level election results info
    # anotehr for district level candidate results info
    results = dict() 
    results_candidates = dict() 
    results_list = list() 
    results_candidates_list = list() 
    for guid in guid_list:
        # the idea of using yaml comes from StackOverflow
        # was not able to convert string to dict; yaml solved the issue
        # url: https://stackoverflow.com/questions/988228/
        #      convert-a-string-representation-of-a-dictionary-to-a-dictionary
        # author: color-blind
        results = dict(yaml.load(guid))
        for candidate_info in results['candidates']:
            candidate_info.update(seat=results['seat'], state_id=results['state_id'])
            results_candidates_list.append(candidate_info)
        results_list.append(results)

    return pd.DataFrame(results_list), pd.DataFrame(results_candidates_list)

if __name__ == "__main__":
    
    election_results, candidate_results = get_results("https://www.nytimes.com/elections/results/house")

    election_results.to_csv('election_results_raw.csv', index=False)
    candidate_results.to_csv('candidate_results_raw.csv', index=False)

