import requests
import re
import yaml
import pandas as pd
import numpy as np
import math

#------------------------------------------------------------------------------------
# get imputed votes 
def get_imputed_votes(source, state_id):                                                       
     return source[source['state_id']==state_id]['votes'].values[0] 

#------------------------------------------------------------------------------------
# get number of seats in state 
def get_state_seat_count(df, state):                                                       
     return df[(df['state_id']==state) & (df['winner']==True)].winner.size

#------------------------------------------------------------------------------------
# get district total vote count 
def get_district_vote_count(df, state_id_seat):
     state_id = state_id_seat[0]
     seat = state_id_seat[1]
     return df[(df['state_id']==state_id) & (df['seat']==seat)]['votes'].sum() 

#------------------------------------------------------------------------------------
# get state total vote count 
def get_state_vote_count(df, state_id):
     return df[df['state_id']==state_id]['votes'].sum() 

#------------------------------------------------------------------------------------
# get district total vote count 
def get_district_candidate_count(df, state_id_seat):
     state_id = state_id_seat[0]
     seat = state_id_seat[1]
     return df[(df['state_id']==state_id) & (df['seat']==seat)]['votes'].size

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
    # for losers, it's everything 
    # for winners, is number of votes above 50% + 1 of total votes
    df2['wasted_votes'] = (np.where(df2['winner']!=True, df2['votes'],\
         df2['votes'] - (df2['district_vote_count']//2 + 1))).astype(int)

    df2['winner2'] = np.where(df2['winner']==True,1,0)
    df2['contested'] = np.where(df2['percent']==100,1,0)
    df2['party_id2'] = np.where((df2['party_id']=='democrat') | (df2['party_id']=='republican'),\
                                 df2['party_id'],'other')
    
    return df2
    
if __name__ == "__main__":

    candidate_results_raw = pd.read_csv('candidate_results_raw.csv')
    
    candidate_results_clean = clean_candidate_results(candidate_results_raw)

    candidate_results_clean.to_csv('candidate_result_clean.csv', index=False)

