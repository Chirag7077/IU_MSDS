import os
import sys
import requests
import re
import yaml
import pandas as pd
import numpy as np
import math

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

#------------------------------------------------------------------------------------
# get nyt election results 
#
def get_nyt_election_results():

    url_data_source = "https://www.nytimes.com/elections/results/house"
    fname_election_results  = 'data/election_results_raw.csv' 
    fname_candidate_results = 'data/candidate_results_raw.csv' 
    fname_clean_results = 'data/candidate_results_clean.csv'

    if not os.path.exists(fname_election_results):
        print('Downloading election results to data directory...')
        election_results, candidate_results = get_results(url_data_source)
        election_results.to_csv(fname_election_results, index=False)
        candidate_results.to_csv(fname_candidate_results, index=False)
        print('Cleaning data...')
        candidate_results_raw = pd.read_csv('data/candidate_results_raw.csv')
        candidate_results_clean = clean_candidate_results(candidate_results_raw)
        candidate_results_clean.to_csv('data/candidate_results_clean.csv', index=False)
    else:
        print("Retrieving clean data ('{}')".format(fname_clean_results))
        candidate_results_clean = pd.read_csv(fname_clean_results)

    print("Done!")
    return candidate_results_clean

#------------------------------------------------------------------------------------
# get imputed votes 
#
def get_imputed_votes(source, state_id):                                                       
    return source[source['state_id']==state_id]['votes'].values[0] 

#------------------------------------------------------------------------------------
# get number of seats in state 
#
def get_state_seat_count(df, state):                                                       
    return df[(df['state_id']==state) & (df['winner']==True)].winner.size

#------------------------------------------------------------------------------------
# get district total vote count 
#
def get_district_vote_count(df, state_id_seat):
    state_id = state_id_seat[0]
    seat = state_id_seat[1]
    return df[(df['state_id']==state_id) & (df['seat']==seat)]['votes'].sum() 

#------------------------------------------------------------------------------------
# get state total vote count 
#
def get_state_vote_count(df, state_id):
    return df[df['state_id']==state_id]['votes'].sum() 

#------------------------------------------------------------------------------------
# get district total vote count 
#
def get_district_candidate_count(df, state_id_seat):
    state_id = state_id_seat[0]
    seat = state_id_seat[1]
    filter = (df['state_id']==state_id) & (df['seat']==seat) & (df['candidate_key']!='dummy')
    return df[filter]['votes'].size

#------------------------------------------------------------------------------------
# get winner wasted votes 
#
def get_winner_wasted_votes(df, row_key):

    state_id         = row_key[0]
    seat             = row_key[1]
    candidate_id     = row_key[2]

    # dummy row cannot be winner
    if candidate_id == 'dummy': return 0

    filter = (df['state_id']==state_id) & (df['seat']==seat) & (df['candidate_id']==candidate_id)
    data_row = df[filter].to_dict(orient='list')

    winner = data_row['winner'][0] 
    district_candidate_count = data_row['district_candidate_count'][0]
    district_votes_count = data_row['district_vote_count'][0]
    candidate_votes_count = data_row['votes'][0]
 
    if not winner: return 0

    # winning votes calculation:
    #     For districts with one uncontested candidates: 
    #         half the votes from uncontested candidate minus 1 are considered wasted votes <br>
    #     For districts with two candidates: 
    #         all votes from winner beyond simple majority are considered wasted votes 
    #     For districts with more than two candidates: 
    #         all votes from winner beyond second most voted candidate plus 1, are wasted votes
  
    if district_candidate_count < 3:
  
        if district_votes_count%2 == 1:
            simple_majority = math.ceil(district_votes_count/2)
        else:
            simple_majority = district_votes_count/2 + 1
        return candidate_votes_count - simple_majority 

    else:
 
        filter = (df['state_id']==state_id) & (df['seat']==seat) & (df['winner']==False)
        top_opponent_votes_count = df[filter][['votes']].\
                                   sort_values('votes', ascending=False).values[0][0]
        return candidate_votes_count - (top_opponent_votes_count + 1)


#------------------------------------------------------------------------------------
# clean candidate results 
# in: candidate results dataframe
# out: original dataframe with the following changes
#      incumbent = NaN changed to False
#      winner = NaN changed to False
#      percent = 0 changed to 100.0
#      percent_display =  0 changed to 100.0
#      votes = 0 changed to imputed value:average votes for districts in state 
# Note: percent_display = votes = 0 means election was uncontested OR unopposed
#
def clean_candidate_results(df):

    df.loc[df['incumbent'].isnull(), 'incumbent'] = False
    df.loc[df['winner'].isnull(), 'winner'] = False
    df.loc[df['percent']==0.0, 'percent'] = 100.0 
    df.loc[df['percent_display']==0.0, 'percent_display'] = 100.0 

    # create fields to facilitate counts
    df['winner2'] = np.where(df['winner']==True,1,0)
    df['uncontested'] = np.where(df['percent']==100,1,0)

    # create field to facilitate grouping of votes into three parties only
    df['party_id2'] = np.where((df['party_id']=='democrat') | (df['party_id']=='republican'),\
                               df['party_id'],'other')

    # identify missing parties for all seats 
    # we want every state/seat combination to have a row for a republican, a democrat and an "other"
    missing = list()                                                     
    for index, row in df[['state_id','seat']].drop_duplicates().iterrows():
        state_id = row['state_id']
        seat = row['seat']
        rs = df[(df['state_id']==state_id) & (df['seat']==seat) & (df['party_id2']=='republican')].size
        ds = df[(df['state_id']==state_id) & (df['seat']==seat) & (df['party_id2']=='democrat')].size
        os = df[(df['state_id']==state_id) & (df['seat']==seat) & (df['party_id2']=='other')].size
        if rs == 0: missing.append((state_id, seat, 'republican'))
        if ds == 0: missing.append((state_id, seat, 'democrat'))
        if os == 0: missing.append((state_id, seat, 'other'))

    # create dataset to concat to df with all rows missing from df 
    # fields in dataset
    # 'candidate_id', 'candidate_key', 'first_name', 'incumbent', 'last_name',
    # 'name_display', 'order', 'party_id', 'percent', 'percent_display',
    # 'seat', 'state_id', 'votes', 'winner', 'winner2', 'uncontested',
    # 'party_id2'
    missing_to_add_to_df = list()
    for state_id, seat, party_id2 in missing:
        missing_to_add_to_df.append(('dummy', 'dummy', 'dummy', False, 'dummy',\
                                     'dummy', 0, 'dummy', 0, 0,\
                                     seat, state_id, 0, False, 0, 0, party_id2))
    df_missing_to_add_to_df = pd.DataFrame(missing_to_add_to_df)
    df_missing_to_add_to_df.columns = df.columns
    df = pd.concat([df, df_missing_to_add_to_df], axis=0).copy()
    
    # will impute votes for uncontested OR unopposed winners
    # partition dataframeand then contactenate after imputing values
    # partition 1 = contested districts
    df_contested_districts = df[~((df['votes']==0) & (df['winner']==True))].copy()
    # partition 2 = uncontested OR unopposed districts
    df_uncontested_districts = df[((df['votes']==0) & (df['winner']==True))].copy()

    # calculate avg district votes for contested districts for each state
    df_avg_state_seat_votes = df_contested_districts.groupby(['state_id','seat']).agg({'votes': np.sum})
    df_avg_state_seat_votes.reset_index(inplace=True)
    df_avg_state_seat_votes = df_avg_state_seat_votes.groupby('state_id').agg({'votes': np.mean})
    df_avg_state_seat_votes['votes'] = df_avg_state_seat_votes['votes'].astype(int)
    df_avg_state_seat_votes.reset_index(inplace=True)

    # update zero votes in partition 2 dataframe
    df_uncontested_districts['votes'] = df_uncontested_districts['state_id'].\
                     apply(lambda x: get_imputed_votes(df_avg_state_seat_votes,x))     

    # concatenate back the two partitions
    df2 = pd.concat([df_contested_districts, df_uncontested_districts], axis=0) 
    df2.sort_values(['state_id','seat'], inplace=True)

    # create field to track number of seat count for state 
    df2['state_seat_count'] = df2['state_id'].\
                     apply(lambda x: get_state_seat_count(df2, x))     

    # create field to track total number of votes for district 
    df2['district_vote_count'] = df2[['state_id','seat']].\
                     apply(lambda x: get_district_vote_count(df2, x), axis=1)     

    # create field to track total number of votes for state 
    df2['state_vote_count'] = df2['state_id'].\
                     apply(lambda x: get_state_vote_count(df2, x))

    # create field to track total number of candidates in the district 
    df2['district_candidate_count'] = df2[['state_id','seat']].\
                     apply(lambda x: get_district_candidate_count(df2, x), axis=1)     

    # calculate wasted votes
    # votes from losing candidate are considered wasted votes always
    # winning votes calculation:
    #     For districts with two candidates: 
    #         all votes from winner beyond simple majority are considered wasted votes 
    #     For districts with more than two candidates: 
    #         all votes from winner beyond second most voted candidate plus 1, are wasted votes

    df2['wasted_winning_votes'] = df2[['state_id','seat','candidate_id']].\
                     apply(lambda x: get_winner_wasted_votes(df2, x), axis=1)     

    df2['wasted_losing_votes'] = (np.where(df2['winner']!=True, df2['votes'], 0)).astype(int)

    df2['wasted_votes'] = df2['wasted_winning_votes'] + df2['wasted_losing_votes']

    df2.drop(['wasted_winning_votes', 'wasted_losing_votes'], inplace=True)

    return df2

#------------------------------------------------------------------------------------
# get_overall_results
# summarizes results at national level 
# returns:
# Party, Votes, Seats_Won, %_Votes, %_Seats_Won, Votes_to_Elect_Candidate, %_Total_Votes, sort_order 
# 
def get_overall_results(df):
    overall = df.groupby('party_id2').agg({'votes': np.sum, 'winner2': np.sum}).reset_index()
    overall['%votes'] = overall['votes'] / overall.votes.sum() * 100
    overall['%winner2'] = overall['winner2'] / overall.winner2.sum() * 100
    overall['votesperseat'] = overall['votes'] / overall['winner2']
    overall['%totalvotesperseat'] = overall['votesperseat'] / overall.votes.sum() * 100
    col_names = ['Party','Votes','Seats_Won','%_Votes','%_Seats_Won',\
               'Votes_to_Elect_Candidate','%_Total_Votes']
    overall.columns = col_names
    overall['sort_order'] = np.where(overall['Party']=='republican', 1,\
                          np.where(overall['Party']=='democrat', 2, 3))
    overall = overall.sort_values('sort_order', ascending=True)
    return overall 

#------------------------------------------------------------------------------------
# get_results_bystate
# summarizes results at the state level 
# returns:
# 'State', 'Party', 'Votes', 'Seats_Won','State_Seat_Count','State_Vote_Count','%_Votes',
# '%_Winners','Votes_to_Elect_Candidate','%_of_Total'
# 
def get_results_bystate(df):

    def get_missed_votes(row_data):

         # NEED to calculate new column "Missed_Seats"
         # Only states where sum of missed_seats > 0 should be included in analysis of gerrymendering
         
         pct_votes = row_data[0]/100
         state_seat_count = row_data[1]
         seats_won = row_data[2]
         return np.maximum((math.floor(pct_votes * state_seat_count) - seats_won),0)

    bystate = df.groupby(['state_id','party_id2']).agg({'votes': np.sum, 'winner2': np.sum,\
                      'state_seat_count': np.max, 'state_vote_count': np.max}).\
                      reset_index()
    bystate['%votes'] = bystate['votes'] / bystate['state_vote_count'] * 100
    bystate['%winner2'] = bystate['winner2'] / bystate['state_seat_count'] * 100
    bystate['votesperseat'] = bystate['votes'] / bystate['winner2']
    bystate['%statevotesperseat'] = bystate['votesperseat'] / bystate.votes.sum() * 100
    bystate['missed_votes'] = bystate[['%votes','state_seat_count','winner2']].\
                              apply(lambda x: get_missed_votes(x), axis=1)
    col_names2 = ['State','Party','Votes','Seats_Won','State_Seat_Count','State_Vote_Count',\
                  '%_Votes','%_Winners','Votes_to_Elect_Candidate','%_of_Total','Missed_Seats']
    bystate.columns = col_names2
    bystate['sort_order'] = np.where(bystate['Party']=='republican', 1, \
                         np.where(bystate['Party']=='democrat', 2, 3))
    bystate = bystate.sort_values(['State','sort_order'], ascending=True)

    return bystate

#------------------------------------------------------------------------------------
# get wasted votes 
# returns wasted vote totals by state/district/party 
# returns:
#    'state_id', 'seat', 'party_id2', 'candidate_key', 'district_candidate_count',\
#    'votes', 'district_vote_count','state_vote_count','wasted_votes' 
# 
def get_wasted_votes(df):

    cols = ['state_id','seat','party_id2','candidate_key','district_candidate_count','votes',\
            'district_vote_count','state_vote_count','wasted_votes']
    exclude_clause = (df['state_seat_count']>1)
    wasted = df[exclude_clause][cols].\
             sort_values('wasted_votes',ascending=False)

    wasted['state_id_seat'] = wasted[['state_id','seat']].\
                              apply(lambda x: (x[0] + ' / ' + str(x[1])), axis=1)
    return wasted

#------------------------------------------------------------------------------------
# get efficiency gap
# calculates efficiency gap per state using formula
# (total dem wasted votes - total rep wasted votes) / total votes 
# returns:
#     state_id, democrat_wasted_votes, republican_wasted_votes, state_vote_count, efficiency_gap,
#     favored_party, abs_efficiency_gap
# 
def get_efficiency_gap(df, bystate):

    # states with missed seats
    states_with_missed_seats = bystate[bystate['Missed_Seats']>0].State.values.tolist()

    exclude_clause = df['state_seat_count']>1
    filter = (exclude_clause) & (df['party_id2']!='other')
    eg = df[filter].groupby(['state_id','party_id2']).\
                             agg({'wasted_votes': np.sum, 'state_vote_count': np.max}).reset_index()
    eg['democrat_wasted_votes'] = eg['wasted_votes'].shift(1)
    eg['republican_wasted_votes'] = eg['wasted_votes']
    eg = eg[eg['party_id2']=='republican'][['state_id','democrat_wasted_votes',\
                             'republican_wasted_votes','state_vote_count']]
    eg['efficiency_gap'] = 100 * (eg.democrat_wasted_votes - eg.republican_wasted_votes )\
                                  / eg.state_vote_count 
    eg['favored_party'] = np.where(eg.efficiency_gap < 0, 'democrat', 'republican')
    eg['abs_efficiency_gap'] = np.where(eg.efficiency_gap < 0, -1 * eg.efficiency_gap, eg.efficiency_gap)
    eg['missed_seats'] = eg['state_id'].apply(lambda x: (x in states_with_missed_seats)) 
    eg.sort_values('abs_efficiency_gap', ascending=False, inplace=True)
    return eg


if __name__ == "__main__":

    pass


