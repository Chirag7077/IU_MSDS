import os
import sys
import requests
import re
import yaml
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

light_color = cm.Paired(8) 
dark_color = cm.Paired(9) 
republican_red = '#b81800'  # (code from politico.com)
democrat_blue = '#007DD6'   # (dode from politico.com)
other_teal = '#E7A520'      # (dode from politico.com) 
party_color = {'republican': republican_red, 'democrat': democrat_blue, 'other': other_teal}
us_state = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
             'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
             'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 
             'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 
             'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 
             'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 
             'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 
             'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 
             'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 
             'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 
             'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}

#------------------------------------------------------------------------------------
# plot_votes_and_seats_won_by_party
# 
def plot_votes_and_seats_won_by_party(overall, fig, ax):

     N = 3
     ind = np.arange(N)  # the x locations for the groups
     width = 0.4        # the width of the bars
    

     #plt.rcdefaults()
    
     pct_votes = overall['%_Votes'].values
     pct_seats = overall['%_Seats_Won'].values
     rects1 = ax.bar(ind, pct_votes, width, color=light_color)
     rects2 = ax.bar(ind + width, pct_seats, width, color=dark_color)
    
     # add some text for labels, title and axes ticks
     ax.axes.get_yaxis().set_visible(False)
     ax.set_xlabel('\nElector')
     ax.set_title('')
     ax.set_xticks(ind + width / 2)
     ax.set_xticklabels(('Republican', 'Democrat', 'Others'))
    
     ax.legend((rects1[0], rects2[0]), ('Votes', 'Seats Won'))
    
     def autolabel(rects):
         # place label on bars
         for rect in rects:
             height = rect.get_height()
             ax.text(rect.get_x() + rect.get_width()/2., 1.001*height, \
                    str(round(height,1)) + "%", ha='center', va='bottom')
     autolabel(rects1)
     autolabel(rects2)
    
     plt.title('Votes and Seats Won by Party')


#------------------------------------------------------------------------------------
# plot_when_less_is_more_national
# 
def plot_when_less_is_more_national(overall):

    republican_usa_avg_votes_per_seat = overall[overall['Party']=='republican']['%_Total_Votes'].values
    democrat_usa_avg_votes_per_seat = overall[overall['Party']=='democrat']['%_Total_Votes'].values
    #plt.rcdefaults()
    plt.figure(figsize=(8,1.7))
    parties = ('Democrat','Republican')
    y_pos = np.arange(len(parties))
    pct_votes = [democrat_usa_avg_votes_per_seat, republican_usa_avg_votes_per_seat]
    
    plt.barh(y_pos, pct_votes, align='center', color=[democrat_blue, republican_red])
    plt.yticks(y_pos, parties)
    plt.xlabel('Percent of Total Votes')
    plt.ylabel('Party')
    plt.title('When Less is More:\nPercent of Votes (Total US) Needed to Elect One House Representative')

#------------------------------------------------------------------------------------
# plot_when_less_is_more_by_state
#
def plot_when_less_is_more_by_state(df, fig, ax):

    states = df['State'].drop_duplicates().values
    rep_filter = df['Party']=='republican'
    dem_filter = df['Party']=='democrat'

    # will sort df, by highest % of total to smalest, regardless of party
    # first zero out numbers for states where votes yielded no wins 
    # (for the purpose of how many votes it took to win seat, infite % == 0 votes)
    df.loc[df['%_of_Total'].isnull(), '%_of_Total'] = 0
    df['%_of_Total'] = np.where(df['%_of_Total']>100, 0, df['%_of_Total'])
    df_rep = df[rep_filter][['State','Party','%_of_Total']]
    df_dem = df[dem_filter][['State','Party','%_of_Total']]
    df_all = pd.merge(df_rep, df_dem, on='State')
    df_all['%_High'] = df_all[['%_of_Total_x','%_of_Total_y']].\
                                apply(lambda x: max(x[0],x[1]), axis=1)
    df_all.sort_values(['%_High'], ascending=False, inplace=True)

    states = df_all['State'].values
    rpct = df_all['%_of_Total_x'].values
    dpct = df_all['%_of_Total_y'].values

    state_count = len(states)
    ind = np.arange(state_count) + 1 # the x locations for the groups
    width = 0.4                      # the width of the bars

    rects1 = ax.bar(ind, rpct, width, color=party_color['republican'])
    rects2 = ax.bar(ind + width, dpct, width, color=party_color['democrat'])

    huge_font = 24
    large_font = 20
    small_font = 16
         
    ax.set_xticks(ind + width/2)

    ax.legend((rects1[0], rects2[0]), ('Republican', 'Democrat'), fontsize=small_font, loc='upper right')

    # add some text for labels, title and axes ticks
    ax.axes.get_yaxis().set_visible(True)
    ax.axes.set_xlim(0.5,51)
    ax.set_xlabel('State', size=large_font)
    ax.set_ylabel('% of National Votes', size=large_font)
  
    ax.set_xticklabels(states, size=small_font)
    ax.set_yticklabels(np.arange(0,0.9,0.1), size=small_font)

    plot_title = 'Percent of National Votes Needed to Elect Party Candidate (Less is More)'
    ax.set_title(plot_title, size=huge_font)


#------------------------------------------------------------------------------------
def plot_when_less_is_more_by_state_party(overall, bystate, party_id, fig, ax):

    avg_votes_per_seat = overall[overall['Party']==party_id]['%_Total_Votes'].values
    filter = ((bystate['Party']==party_id) & (bystate['%_of_Total']<101) & \
              (bystate['%_of_Total']>0))
    df = bystate[filter][['State','%_of_Total']].\
                 sort_values(['%_of_Total','State'], ascending=[False, True])


    ax.set_xlabel('State')
    ax.set_ylabel('% of National Votes')
    ax.axhline(y=avg_votes_per_seat, linestyle='-', alpha=0.2, color=party_color[party_id])
    plt_title = 'Percent of National Votes Needed to Elect ' + party_id.title() + \
                ' in State (Less Votes = More Power)'
    ax.set_title(plt_title)
    df.plot.bar(x='State', ax=ax, color=party_color[party_id], legend=None, ylim=(0,0.8))


#------------------------------------------------------------------------------------
# plot_when_less_is_more_state_detail 
# 
def plot_when_less_is_more_state_detail(df, fig, ax, figtitle, *states):

    # define functions to analyse state level data
    def get_state_data(df, state_id):
        # checking Alaska results
        state_filter = (df['state_id']==state_id)
        number_of_seats = df[state_filter].seat.values.max()
        df_state = df[state_filter].groupby('party_id2').\
                                    agg({'votes': np.sum, 'winner2': np.sum}).reset_index()
        df_state['%votes'] = df_state['votes'] / df_state.votes.sum() * 100
        df_state['%winner2'] = df_state['winner2'] / df_state.winner2.sum() * 100
        df_state['votesperseat'] = df_state['votes']/df_state['winner2']
        df_state.columns = ['Party','Votes','Seats_Won','%_Votes','%_Seats_Won',\
                            'Votes_to_Elect_Candidate']
        df_state['sort_order'] = np.where(df_state['Party']=='republican', 1, \
                                 np.where(df_state['Party']=='democrat', 2, 3))
        df_state = df_state.sort_values('sort_order', ascending=True)
        return df_state, number_of_seats
    
    def cast_plot(fig, ax, df, state_name, seat_count):
        pct_votes = df['%_Votes'].values
        pct_seats_won = df['%_Seats_Won'].values
        N = len(pct_votes)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.4        # the width of the bars    
        
        rects1 = ax.bar(ind, pct_votes, width, color=light_color)
        rects2 = ax.bar(ind + width, pct_seats_won, width, color=dark_color)
    
        # add some text for labels, title and axes ticks
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.set_ylim(0,105)
        ax.set_xlabel('\nElector')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('Republican', 'Democrat', 'Others'))
    
        ax.legend((rects1[0], rects2[0]), ('Votes', 'Seats Won'))
    
        def autolabel(rects):
            # add label to bar
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.001*height,
                    str(round(height,1)) + "%", ha='center', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
        
        plot_title = state_name + ' (' + str(seat_count) + ' seats)'
        ax.set_title(plot_title)
    
    if len(states) > 1:

        if len(states) != len(ax):
            raise ValueError('Number of plots must match number of states')
    
        # traverse the list of states received in top function call
        i = 0
        for plot_state in states:
            # retrieve rows for state plot and count of state seats
            df_state, state_seat_cnt = get_state_data(df, plot_state)
            # plot the state 
            cast_plot(fig, ax[i], df_state, us_state[plot_state], state_seat_cnt)
            i += 1

    else:
            # retrieve rows for state plot and count of state seats
            df_state, state_seat_cnt = get_state_data(df, states[0])
            # plot the single state 
            cast_plot(fig, ax, df_state, us_state[states[0]], state_seat_cnt)
    
    # set title of figure
    fig.suptitle(figtitle, fontsize=14)
     
#------------------------------------------------------------------------------------
# plot_state_votes_by_seat 
# 
def plot_state_votes_by_seat(df, state_id, fig, ax):

    # define functions to analyse state level data
    def get_votes_by_state_seat_party(df, state_id):
        # checking Alaska results
        state_filter = (df['state_id']==state_id)
        df_state = df[state_filter].groupby(('state_id','seat','party_id2')).\
                                              agg({'votes': np.sum}).reset_index()
        df_state = df_state[['state_id','seat','party_id2','votes']]
        df_state.columns = ['State','Seat','Party','Votes']
        df_state['sort_order'] = np.where(df_state['Party']=='republican', 1, \
                                 np.where(df_state['Party']=='democrat', 2, 3))
        df_state = df_state.sort_values(['Seat', 'sort_order'], ascending=[True, True])
        return df_state

    def cast_plot(df, state_id, fig, ax):
         
        seats = df['Seat'].drop_duplicates().values
        rep_filter = df['Party']=='republican'
        dem_filter = df['Party']=='democrat'
        oth_filter = df['Party']=='other'
        rvotes = df[rep_filter]['Votes'].values
        dvotes = df[dem_filter]['Votes'].values
        ovotes = df[oth_filter]['Votes'].values
     
        seat_count = len(seats)
        ind = np.arange(seat_count) * 2   # the x locations for the groups
        width = 0.5                       # the width of the bars

        rects1 = ax.bar(ind, rvotes, width, color=party_color['republican'])
        rects2 = ax.bar(ind + width, dvotes, width, color=party_color['democrat'])
        rects3 = ax.bar(ind + 2*width, ovotes, width, color=party_color['other'])
             
        ax.set_xticks( ind + width)
        
        if ovotes.sum() > 0:
           ax.legend((rects1[0], rects2[0], rects3[0]), ('Republican', 'Democrat','Other'),\
                      bbox_to_anchor=(1.01, 1), loc=2)
        else: 
           ax.legend((rects1[0], rects2[0]), ('Republican', 'Democrat'),\
                      bbox_to_anchor=(1.01, 1), loc=2)

        # add some text for labels, title and axes ticks
        ax.axes.get_yaxis().set_visible(True)
        #ax.axes.set_ylim(0,100)
        ax.set_xlabel('Seat Number')
        ax.set_ylabel('Number of Votes')
     
        ax.set_xticklabels(seats)

        plot_title = '{} Votes - {} Seats'.format(us_state[state_id], seat_count)
        ax.set_title(plot_title)

    df_state = get_votes_by_state_seat_party(df, state_id)
    cast_plot(df_state, state_id, fig, ax)
    

#------------------------------------------------------------------------------------
# plot_wasted_votes
# 
def plot_wasted_votes(df, fig, ax):
    x = df.wasted_votes.values[0:25]
    y = df.state_id_seat.values[0:25]
    party_id2 = df.party_id2.values[0:25]
    y_pos = np.arange(len(y)) + 0.5
    
    ax.barh(y_pos, x, align='center',color='green')
    i = 0
    for party in party_id2:
        ax.get_children()[i].set_color(party_color[party])
        i += 1
    ax.set_yticks(y_pos)
    ax.set_ylim(0, 25)
    ax.set_yticklabels(y, size=8)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.xticks(fontsize=8)
    ax.set_ylabel('State / District', size=12)
    ax.set_xlabel('Number of Wasted Votes',size=12)
    ax.set_title('Wasted Vote by State / District', size=14)
    rep_patch = mpatches.Patch(color=party_color['republican'], label='Republican')
    dem_patch = mpatches.Patch(color=party_color['democrat'], label='Democrat')
    plt.legend(handles=[rep_patch, dem_patch], fontsize=14, bbox_to_anchor=(1.01, 1), loc=2)

#------------------------------------------------------------------------------------
# plot efficiency gap
# 
def plot_efficiency_gap(df, fig, ax):

    # remove states with no missed seats
    df = df[df['missed_seats']]

    x = df.abs_efficiency_gap.values
    y = df.state_id.values
    party_id2 = df.favored_party.values
    y_pos = np.arange(len(y)) + 0.5
    
    ax.barh(y_pos, x, align='center')
    i = 0
    for party in party_id2:
        ax.get_children()[i].set_color(party_color[party])
        i += 1
    ax.set_yticks(y_pos)
    ax.set_ylim(0, len(y))
    ax.set_yticklabels(y, size=12)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.xticks(fontsize=8)
    ax.xaxis.grid()
    ax.set_ylabel('State', size=12)
    ax.set_xlabel('% Efficiency Gap',size=12)
    ax.set_title('% Efficiency Gap by State', size=14)
    rep_patch = mpatches.Patch(color=party_color['republican'], label='Republican')
    dem_patch = mpatches.Patch(color=party_color['democrat'], label='Democrat')
    plt.legend(handles=[rep_patch, dem_patch], fontsize=14, title='Favored Party',\
               bbox_to_anchor=(1.01, 1), loc=2)


if __name__ == "__main__":

     pass

