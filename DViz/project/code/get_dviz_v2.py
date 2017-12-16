import os
import sys
import requests
import re
import yaml
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from mpl_toolkits.basemap import Basemap
from matplotlib import cm

light_color = cm.Paired(8) 
dark_color = cm.Paired(9) 
republican_red = '#b81800'  # (code from politico.com)
republican_dark = '#800000' # (code from http://www.color-hex.com/)
democrat_blue = '#007DD6'   # (dode from politico.com)
democrat_dark = '#232066'
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
# state fips number - used to identify state read in shapefile.dbf
fpstate = {'01': 'AL', '02': 'AK', '60': 'AS', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT',
           '10': 'DE', '11': 'DC', '12': 'FL', '64': 'FM', '13': 'GA', '66': 'GU', '15': 'HI', '16': 'ID',
           '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '68': 'MH',
           '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE',
           '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '69': 'MP',
           '39': 'OH', '40': 'OK', '41': 'OR', '70': 'PW', '42': 'PA', '72': 'PR', '44': 'RI', '45': 'SC',
           '46': 'SD', '47': 'TN', '48': 'TX', '74': 'UM', '49': 'UT', '50': 'VT', '51': 'VA', '78': 'VI',
           '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'}
statefp = {'AL': '01', 'AK': '02', 'AS': '60', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09',
           'DE': '10', 'DC': '11', 'FL': '12', 'FM': '64', 'GA': '13', 'GU': '66', 'HI': '15', 'ID': '16',
           'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MH': '68',
           'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31',
           'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'MP': '69',
           'OH': '39', 'OK': '40', 'OR': '41', 'PW': '70', 'PA': '42', 'PR': '72', 'RI': '44', 'SC': '45',
           'SD': '46', 'TN': '47', 'TX': '48', 'UM': '74', 'UT': '49', 'VT': '50', 'VA': '51', 'VI': '78',
           'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56'}
# state coordinates to create map: llcrnrlon, urcrnrlon, urcrnrlat, llcrnrlat
state_coordinates = {'AL': (-89.00,-84.37,35.50,29.50), 'AK': (173.50,-130.00,71.50,51.25),
                     'AZ': (-115.87,-107.00,38.00,30.33), 'AR': (-94.62,-89.62,36.50,33.00),
                     'CA': (-125.00,-112.80,42.70,31.53), 'CO': (-109.92,-101.00,41.50,36.50),
                     'CT': (-73.95,-71.55,42.15,40.90), 'DE': (-75.80,-75.00,39.85,38.45),
                     'DC': (-77.12,-76.87,39.00,38.87), 'FL': (-88.40,-79.00,31.50,24.0),
                     'GA': (-86.22,-80.05,35.50,29.75), 'HI': (-160.25,-154.75,22.23,18.87),
                     'ID': (-117.25,-111.00,49.00,42.00), 'IL': (-92.00,-86.50,43.00,36.50),
                     'IN': (-88.72,-83.95,42.25,37.27), 'IA': (-96.62,-90.12,43.50,40.37),
                     'KS': (-103.20,-93.28,40.70,36.20), 'KY': (-89.58,-81.95,39.15,36.62),
                     'LA': (-94.05,-88.82,33.02,28.92), 'ME': (-71.13,-66.88,47.47,42.97),
                     'MD': (-80.00,-74.50,40.15,37.57), 'MA': (-73.92,-69.52,43.07,41.02),
                     'MI': (-90.50,-81.37,48.88,40.90), 'MN': (-97.25,-89.50,49.38,43.50),
                     'MS': (-91.63,-88.12,35.00,30.00), 'MO': (-96.48,-88.10,41.22,35.20), 
                     'MT': (-116.05,-104.03,49.00,44.37), 'NE': (-104.55,-94.32,43.70,39.20),
                     'NV': (-120.00,-114.05,42.00,35.00), 'NH': (-72.57,-70.58,45.35,42.70),
                     'NJ': (-75.55,-73.87,41.37,38.92), 'NM': (-109.05,-103.00,37.00,31.33),
                     'NY': (-79.77,-71.87,45.02,40.50), 'NC': (-85.03,-74.82,37.60,32.85),
                     'ND': (-104.05,-96.55,49.00,45.93), 'OH': (-85.52,-79.72,42.30,37.90),
                     'OK': (-103.50,-93.53,37.60,33.02), 'OR': (-124.58,-116.45,46.27,41.90),
                     'PA': (-81.02,-74.18,42.57,39.22), 'PR': (-67.95,-65.22,18.53,17.92),
                     'RI': (-71.92,-71.12,42.02,41.13), 'SC': (-83.67,-78.02,35.72,31.50),
                     'SD': (-104.05,-96.43,45.93,42.48), 'TN': (-90.82,-81.13,37.58,33.97),
                     'TX': (-107.3,-91.70,37.50,24.68), 'UT': (-115.05,-107.85,42.50,36.50),
                     'VT': (-73.60,-71.47,45.00,42.72), 'VA': (-84.8,-74.1,40.00,35.60),
                     'VI': (-83.68,-75.25,39.47,36.53), 'WA': (-124.77,-116.92,49.00,45.53),
                     'WV': (-82.65,-77.73,40.63,37.20), 'WI': (-93.50,-85.85,47.42,41.80), 
                     'WY': (-111.10,-104.00,45.00,41.00)}
                     
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

    #filter = (df_all['%_Total_Votes']<=1)
    #avg_votes_per_seat = df[filter]['%_Total_Votes'].values.mean()

    states = df_all['State'].values
    rpct = df_all['%_of_Total_x'].values
    dpct = df_all['%_of_Total_y'].values

    state_count = len(states)
    ind = np.arange(state_count) + 1 # the x locations for the groups
    width = 0.4                      # the width of the bars

    rects1 = ax.bar(ind, rpct, width, color=party_color['republican'])
    rects2 = ax.bar(ind + width, dpct, width, color=party_color['democrat'])
    #ax.axhline(y=avg_votes_per_seat, linestyle='-', alpha=0.2, color='k')

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

    #avg_votes_per_seat = overall[overall['Party']==party_id]['%_Total_Votes'].values
    filter = (overall['%_Total_Votes']<=1)
    avg_votes_per_seat = overall[filter]['%_Total_Votes'].values.mean()
    #print(avg_votes_per_seat)
    filter = ((bystate['Party']==party_id) & (bystate['%_of_Total']<101) & \
              (bystate['%_of_Total']>0))
    df = bystate[filter][['State','%_of_Total']].\
                 sort_values(['%_of_Total','State'], ascending=[False, True])

    ax.set_xlabel('State')
    ax.set_ylabel('% of National Votes')
    #ax.axhline(y=avg_votes_per_seat, linestyle='-', alpha=0.2, color=party_color[party_id])
    ax.axhline(y=avg_votes_per_seat, linestyle='-', alpha=0.2, color='k')
    plt_title = 'Percent of National Votes Needed to Elect ' + party_id.title() + \
                ' in State (Less Votes = More Power)'
    ax.set_title(plt_title, size=14)
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
def plot_state_votes_by_seat(df, state_id, ax, solo=False):

    # define functions to analyse state level data
    def get_votes_by_state_seat_party(df, state_id):

        state_filter = (df['state_id']==state_id)
        df_state = df[state_filter].groupby(('state_id','seat','party_id2')).\
                                              agg({'votes': np.sum}).reset_index()
        df_state = df_state[['state_id','seat','party_id2','votes']]
        df_state.columns = ['State','Seat','Party','Votes']
        df_state['sort_order'] = np.where(df_state['Party']=='republican', 1, \
                                 np.where(df_state['Party']=='democrat', 2, 3))
        df_state = df_state.sort_values(['Seat', 'sort_order'], ascending=[True, True])
        return df_state

    def cast_plot(df, state_id, ax):
         
        seats = df['Seat'].drop_duplicates().values
        rep_filter = df['Party']=='republican'
        dem_filter = df['Party']=='democrat'
        oth_filter = df['Party']=='other'
        rvotes = df[rep_filter]['Votes'].values
        dvotes = df[dem_filter]['Votes'].values
        ovotes = df[oth_filter]['Votes'].values
     
        seat_count = len(seats)
        ind = np.arange(seat_count) * 1.6  # the x locations for the groups
        width = 0.5                        # the width of the bars

        rects1 = ax.bar(ind, rvotes, width, color=party_color['republican'])
        rects2 = ax.bar(ind + width, dvotes, width, color=party_color['democrat'])
        rects3 = ax.bar(ind + 2*width, ovotes, width, color=party_color['other'])
             
        ax.set_xticks( ind + width )
  
        # add some text for labels, title and axes ticks
        ax.axes.get_yaxis().set_visible(True)
        ax.set_xlabel('Seat Number', fontsize=14)
        ax.set_ylabel('Votes', fontsize=14)
    
        if seat_count > 50: 
            ax.set_xticklabels(seats, fontsize=7)
        else:
            ax.set_xticklabels(seats, fontsize=12)

        if solo:      
            
            if ovotes.sum() > 0:
                ax.legend((rects1[0], rects2[0], rects3[0]), ('Republican', 'Democrat','Other'),\
                           bbox_to_anchor=(1.01, 1), loc=2)
            else: 
                ax.legend((rects1[0], rects2[0]), ('Republican', 'Democrat'),\
                           bbox_to_anchor=(1.01, 1), loc=2)

            plot_title = '{} Votes - {} Seats'.format(us_state[state_id], seat_count)

            ax.set_title(plot_title, fontsize=14)

    df_state = get_votes_by_state_seat_party(df, state_id)
    cast_plot(df_state, state_id, ax)
    

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
def plot_efficiency_gap(df, fig, ax, savefile=False):

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

    if savefile:
        plt.tight_layout()
        plt.savefig('images/efficiency_gap.png')

#------------------------------------------------------------------------------------
# plot efficiency gap in terms of "lost" seats
# 
def plot_efficiency_gap_seat(df, ax, savefile=False):

    # remove states with no missed seats
    df = df[df['Missed_Seats']>0].sort_values(['Missed_Seats','State'], ascending=[False, True])

    x = df.Missed_Seats.values
    y = df.State.values
    party_id2 = df.Party.values
    y_pos = np.arange(len(y)) + 0.5 

    ax.barh(y_pos, x, align='center')
    i = 0
    for party in party_id2:
        if party == 'democrat':
            favored_party = 'republican'
        else: 
            favored_party = 'democrat'
        ax.get_children()[i].set_color(party_color[favored_party])
        i += 1

    ax.set_yticks(y_pos)
    ax.set_ylim(0, len(y))
    ax.set_yticklabels(y, size=12)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xticks([0,1,2,3,4])
    plt.xticks(fontsize=8)
    ax.xaxis.grid()

    ax.set_ylabel('State', size=14)
    ax.set_xlabel('Number of Lost Seats',size=14)
    ax.set_title('States with Lost Seats', size=14)
    rep_patch = mpatches.Patch(color=party_color['republican'], label='Republican')
    dem_patch = mpatches.Patch(color=party_color['democrat'], label='Democrat')

    plt.legend(handles=[rep_patch, dem_patch], fontsize=14, title='Favored Party',\
               bbox_to_anchor=(1.01, 1), loc=2)

    if savefile:
        plt.tight_layout()
        plt.savefig('images/efficiency_gap_seat.png')

#------------------------------------------------------------------------------------
# plot efficiency gap in terms of "lost" seats as a percentual of total state seats
# 
def plot_efficiency_gap_seat_percent(df, ax, savefile=False):

    # remove states with no missed seats
    df = df[df['Missed_Seats']>0]  
    df['Missed_Seats_Pct'] = df.Missed_Seats / df.State_Seat_Count * 100
    df = df.sort_values(['Missed_Seats_Pct','State'], ascending=[False, True])
    party_id2 = df.Party.values

    x = df.Missed_Seats_Pct.values
    y = df.State.values

    percents = df.Missed_Seats.values / df.State_Seat_Count.values * 100
    states   = df.State.values

    df = pd.DataFrame(states, percents)
    df.reset_index(inplace=True)
    df.columns = ['percents', 'states']
    df = df.sort_values(['percents','states'], ascending=[False, True])

    y_pos = np.arange(len(y)) + 0.5 

    ax.barh(y_pos, x, align='center')

    i = 0
    for party in party_id2:
        if party == 'democrat':
            favored_party = 'republican'
        else: 
            favored_party = 'democrat'
        ax.get_children()[i].set_color(party_color[favored_party])
        i += 1

    ax.set_yticks(y_pos)
    ax.set_ylim(0, len(y))
    ax.set_yticklabels(y, size=12)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xticks(np.linspace(0,30,7))
    plt.xticks(fontsize=8)
    ax.xaxis.grid()

    ax.set_ylabel('State', size=14)
    ax.set_xlabel('Percentage of Lost Seats',size=14)
    ax.set_title('States with Lost Seats\nPercentage Relative to Total Number of Seats in State', size=14)
    rep_patch = mpatches.Patch(color=party_color['republican'], label='Republican')
    dem_patch = mpatches.Patch(color=party_color['democrat'], label='Democrat')

    plt.legend(handles=[rep_patch, dem_patch], fontsize=14, title='Favored Party',\
               bbox_to_anchor=(1.01, 1), loc=2)

    if savefile:
        plt.tight_layout()
        plt.savefig('images/efficiency_gap_seat_percent.png')


def plot_missed_seats_by_state(df, state_id, ax, print_title=True, savefile=False):
    
    df = df[df['State']==state_id]
    
    democrat_missed = df[df['Party']=='democrat']['Missed_Seats'].values[0].astype('int')
    republican_missed = df[df['Party']=='republican']['Missed_Seats'].values[0].astype('int')

    democrat_seats = df[df['Party']=='democrat']['Seats_Won'].values
    republican_seats = df[df['Party']=='republican']['Seats_Won'].values
    
    democrat_pct = df[df['Party']=='democrat']['%_Winners'].values
    republican_pct = df[df['Party']=='republican']['%_Winners'].values
    
    democrat_votes = df[df['Party']=='democrat']['%_Votes'].values
    other_votes = df[df['Party']=='other']['%_Votes'].values
    republican_votes = df[df['Party']=='republican']['%_Votes'].values
    democrat_votes = np.round(democrat_votes, 2)
    republican_votes = np.round(republican_votes,2)

    pad = 20
    parties = ('Democrat','Republican')
    y_pos = [0, 0.5]

    ax.barh(0, democrat_votes, height=0.3 ,align='center', color=democrat_blue)
    ax.barh(0, other_votes, height=0.3 , align='center', left=democrat_votes, color=other_teal)
    ax.barh(0, republican_votes, height=0.3 ,align='center', left=democrat_votes+other_votes,\
            color=republican_red)
    
    ax.barh(0.20, democrat_pct, height=0.12 ,align='center', color=democrat_dark)
    ax.barh(0.20, republican_pct, height=0.12 ,align='center', left=democrat_pct, color=republican_dark)

    ax2 = ax.twinx()
    ax3 = ax.twiny()
    ax.set_ylabel('Democrat', size=14)
    ax2.set_ylabel('Republican', size=14)
    ax.set_xlabel('% of Votes', size=14)
    ax3.set_xlabel('Number of Seats Won', size=14)
    
    ax2.set_xlim(0,100)
    
    for ax in [ax, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
    
    # add percent votes to bottom bars
    dem_votes = '{}%'.format(democrat_votes[0])
    ax.text(0.013, 0.0, dem_votes, weight='bold', color='w', size=14)
    rep_votes = '{}%'.format(republican_votes[0])
    ax.text(0.90, 0, rep_votes, weight='bold', color='w', size=14)
    
    # add vote counts to top bars
    
    # democrats
    if democrat_seats == 1:
        dem_seats = '{} Seat'.format(democrat_seats[0])
        xpos = 0.91 
    else:
        dem_seats = '{} Seats'.format(democrat_seats[0])
        xpos = 0.89
    if democrat_seats > 0:
        ax.text(0.013,0.18, dem_seats, weight='bold', color='w', size=14)

    # republicans
    if republican_seats == 1:
        rep_seats = '{} Seat'.format(republican_seats[0])
        xpos = 0.91 
    else:
        rep_seats = '{} Seats'.format(republican_seats[0])    
        xpos = 0.89
    if republican_seats > 0:
        plt.text(xpos, 0.18, rep_seats, weight='bold', color='w', size=14)
    
    # add text for missing seats
    if democrat_missed + republican_missed != 0:
        xpos = democrat_pct/100 - 0.027
        if (democrat_missed):
            plt.text(xpos, 0.19, '–{}  +{}'.format(democrat_missed, democrat_missed),\
                     color='white', weight='bold', size=10)
        else:
            plt.text(xpos, 0.19, '+{}  –{}'.format(republican_missed, republican_missed),\
                     color='white', weight='bold', size=10)
    
    ax.axhline(y=0.14, xmin=0, xmax=1, color='w', linewidth = 1)
            
    if print_title:
        plt.title("{} Election Results\n\n".format(us_state[state_id]), weight='bold', size=16)
    
    if savefile:
        plt.tight_layout()
        plt.savefig('images/{}_votes_seat_count.png'.format(state_id))


def get_district_winner(election_results, state, district):

    filter = ((election_results['state_id']==state)     &\
              (election_results['seat']==int(district)) &\
              (election_results['winner2']==1))
    
    winner_party = election_results[filter]['party_id']

    return party_color[winner_party.values[0]]


def print_district_map(eresults, state, ax, force_seat_label=False):

    llcrnrlon, urcrnrlon, urcrnrlat, llcrnrlat = state_coordinates[state]
    lat_0 = (urcrnrlat + llcrnrlat) / 2
    lon_0 = (llcrnrlon + urcrnrlon) / 2
    map = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                  resolution='i', projection='tmerc', lat_0=lat_0, lon_0=lon_0)
    _ = map.readshapefile('data/cb_2016_us_cd115_5m/cb_2016_us_cd115_5m', 'cb_2016_us_cd115_5m')
    map.drawstates(linewidth=2)
    map.drawcountries(linewidth=2)
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='#ddaa66',lake_color='aqua')
    map.drawcoastlines()

    # if state has more than 20 districts won't annotate seat numbers on map
    state_seat_count = eresults[eresults['state_id']==state]['state_seat_count'].values.max()
    doannotate = (state_seat_count < 20) or force_seat_label 

    previous_district = ""
    first_time = True
    for info, shape in zip(map.cb_2016_us_cd115_5m_info, map.cb_2016_us_cd115_5m):
        if info['STATEFP'] == statefp[state]:
            # plot district boundary section
            x, y = zip(*shape)
            map.plot(x, y, marker=None, color='k', linewidth=0.2)
            # track coordinates for current district 
            # current district
            district = info['CD115FP']
            if first_time:
                first_time = False
                previous_district = district
                xsaved = x
                ysaved = y

            if previous_district != district:
                # fill district polygon
                patches = []
                district_full_shape = []
                _ = [district_full_shape.append(pair) for pair in zip(xsaved, ysaved)]
                patches.append( Polygon(np.array(district_full_shape), True) )
                ax.add_collection(PatchCollection(patches, facecolor= 'm', edgecolor='k',\
                                  linewidths=1., zorder=2))
                _ = ax.add_collection(PatchCollection(patches,\
                                      color=get_district_winner(eresults, state, previous_district),\
                                      edgecolor='k', linewidths=0.2, zorder=2))
                if doannotate:
                    # annotate district number
                    npx = np.array(xsaved)
                    npy = np.array(ysaved)
                    xcenter = npx.min() + ( npx.max()-npx.min() ) / 2
                    ycenter = npy.min() + ( npy.max()-npy.min() ) / 2
                    _ = ax.annotate(str(int(previous_district)), xy=(xcenter, ycenter),\
                                            color='w', fontweight='bold')

                # re-initiate variables with new district info
                previous_district = district
                xsaved = x
                ysaved = y                
            else:
                xsaved = xsaved + x
                ysaved = ysaved + y
            
    # fill district polygon
    patches = []
    district_full_shape = []
    _ = [district_full_shape.append(pair) for pair in zip(xsaved, ysaved)]
    patches.append( Polygon(np.array(district_full_shape), True) )
    ax.add_collection(PatchCollection(patches, facecolor= 'm', edgecolor='k', linewidths=1., zorder=2))
    _ = ax.add_collection(PatchCollection(patches,\
                          color=get_district_winner(eresults, state, previous_district),\
                          edgecolor='k', linewidths=0.2, zorder=2))
    
    if doannotate:
        # annotate last district
        npx = np.array(xsaved)
        npy = np.array(ysaved)
        xcenter = npx.min() + ( npx.max()-npx.min() ) / 2
        ycenter = npy.min() + ( npy.max()-npy.min() ) / 2
        _ = ax.annotate(str(int(previous_district)), xy=(xcenter, ycenter), color='w', fontweight='bold')
        ax.set_title("State Map with Seat Numbers", fontsize=14)
    else:
        ax.set_title("State Map with District Lines", fontsize=14)


def print_state_summary_results(election_results, bystate, state, tofile=False):

    plt.figure(figsize=(15,6))

    ax1 = plt.subplot2grid((9,8), (1,0), rowspan=3, colspan=5)
    _ = plot_missed_seats_by_state(bystate, state, ax1, False)

    ax2 = plt.subplot2grid((9,8), (4,0), rowspan=4, colspan=5)
    _ = plot_state_votes_by_seat(election_results, state, ax2)

    ax3 = plt.subplot2grid((9,8), (1,5), rowspan=7, colspan=4)
    _ = print_district_map(election_results, state, ax3)

    plt.suptitle("{} Election Results".format(us_state[state]), fontweight='bold', fontsize=18)

    plt.tight_layout()
    
    if tofile:
        plotfilename = 'images/state_summary_results_{}.png'.format(state)
        print("Saving {}".format(plotfilename))
        plt.savefig(plotfilename, orientation='landscape')


def get_state_colors(all_states):
    all_colors = list()
    _ = [all_colors.append((name, hex)) for name, hex in colors.cnames.items()]
    
    state_colors = list()
    i = 0
    for state in all_states:
        state_colors.append(all_colors[i][1])
        i += 1

    return state_colors
         
         
def plot_when_less_is_more_scatter(df, fig, ax):

    rep_filter = df['Party']=='republican'
    dem_filter = df['Party']=='democrat'

    # will sort df, by highest % of total to smalest, regardless of party
    # first zero out numbers for states where votes yielded no wins
    # (for the purpose of how many votes it took to win seat, infite % == 0 votes)
    df.loc[df['%_of_Total'].isnull(), '%_of_Total'] = 0
    df['%_of_Total'] = np.where(df['%_of_Total']>100, 0, df['%_of_Total'])
    df = df[df['%_of_Total'] != 0]
    df_rep = df[rep_filter][['State','Party','%_of_Total']]
    df_dem = df[dem_filter][['State','Party','%_of_Total']]
    df_all = pd.merge(df_rep, df_dem, on='State')
    df_all['%_High'] = df_all[['%_of_Total_x','%_of_Total_y']].\
                                apply(lambda x: max(x[0],x[1]), axis=1)
    df_all.sort_values(['State'], ascending=False, inplace=True)

    states = df_all['State'].drop_duplicates().values[:]
    states.sort()
    state_colors = get_state_colors(states)
    state_colors='k'
    rpct = df_all['%_of_Total_x'].values
    dpct = df_all['%_of_Total_y'].values

    ax.scatter(rpct, dpct, s=120, alpha=0.4, edgecolors='k', c=state_colors, label=states)
    ax.plot([0,1], linewidth=0.7, c='k')

    huge_font = 24
    large_font = 20
    small_font = 16
    
    ax.axes.set_xlim(-0.01,0.8)
    ax.axes.set_ylim(-0.01,0.8)
    ax.set_xlabel('% Republican Votes', size=12)
    ax.set_ylabel('% Democrat Votes', size=12)
    
    state_patch = list()
    i = 0
    for state in states:
        #state_patch.append(mpatches.Patch(color=state_colors[i], alpha=0.5, label=state))
        i += 1
                           
    #ax.legend(states, state_colors, fontsize=small_font, loc='upper right')
    #ax.legend(handles=state_patch, ncol=3, fontsize=10, title='State', bbox_to_anchor=(1.01, 1), loc=2)
    #ax.legend()
    
    ax.set_title('Votes Needed to Elect Party Candidate (as % of total US votes)\n\
Showing States Where Both Parties Elected Candidate', size=12)
    #ax.set_title('Votes Needed to Elect Candidate (as % of total US votes)', size=14)
    
    ax.text(0.3, 0.53, '   Republicans\nhave more power', color='k')
    ax.text(0.6, 0.5, '    Democrats\nhave more power', color='k')
    ax.annotate('', xy=(0.45, 0.6), xytext=(0.525, 0.525),
            arrowprops=dict(edgecolor=republican_red, facecolor=republican_red, shrink=0.05))
    ax.annotate('', xy=(0.6, 0.45), xytext=(0.525, 0.525),
            arrowprops=dict(edgecolor=democrat_blue, facecolor=democrat_blue, shrink=0.05))


def plot_efficiency_gap_percent_lost_seats(eff_gap_df, bystate, fig, ax, savefile=False):

    # deal with efficiency gap data first - x axis
    # remove states with no missed seats
    df1= eff_gap_df[eff_gap_df['missed_seats']]
    
    # deal with lost seat data second - y axis
    # remove states with no missed seats
    df2= bystate[bystate['Missed_Seats']>0]
    df2['Missed_Seats_Pct'] = df2.Missed_Seats / df2.State_Seat_Count * 100
    df2 = df2[['State','Missed_Seats_Pct']]
    df2.columns = ['state_id','Missed_Seats_Pct']

    df_final = pd.merge(df2[['state_id','Missed_Seats_Pct']],df1[['state_id','abs_efficiency_gap']])

    x = df_final.abs_efficiency_gap.values
    y = df_final.Missed_Seats_Pct.values
    
    ax.scatter(x, y, s=120, alpha=0.4, edgecolors='k', c='k')
    ax.axes.set_xlim(0,25.5)
    ax.axes.set_ylim(0,30)
    ax.set_xlabel('% Efficiency Gap', size=12)
    ax.set_ylabel('% Lost Seats', size=12)
    
    ax.set_title('% Seats Lost, % Efficiency Gap Correlation', size=14)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #plt.plot(x, intercept + slope*x, 'k', linewidth=0.1, label='fitted line')

    r2 = r_value**2
    r2_txt = 'R-Squared = ' + str(round(r2*100,2)) + '%'
    ax.annotate(r2_txt, xy=(16, 27), fontsize=12, fontweight='bold') #, xycoords='figure points')


def plot_uncontested_seats_percent_lost_seats(election_results, bystate, fig, ax):

    # first handle percentage of uncontested seats
    #
    df1 = election_results[election_results['uncontested']==1].\
        groupby(['state_id','state_seat_count']).\
        agg({'state_seat_count': 'max', 'state_id': 'count'})
        
    df1.drop('state_seat_count', axis=1, inplace=True)
    df1.columns=(['uncontested_seat_count'])
    df1.reset_index(inplace=True)
    df1['percent_uncontested_seat_count'] = df1['uncontested_seat_count'] / df1['state_seat_count'] * 100
    
    # get percent of lost seats for all states that had missed seats
    df2 = bystate[bystate['Missed_Seats']>0]
    df2['Missed_Seats_Pct'] = df2.Missed_Seats / df2.State_Seat_Count * 100
    df2 = df2[['State','Missed_Seats_Pct']]
    df2.columns = ['state_id','Missed_Seats_Pct']

    df_final = pd.merge(df2[['state_id','Missed_Seats_Pct']],df1[['state_id','percent_uncontested_seat_count']])

    x = df_final.percent_uncontested_seat_count.values
    #x.sort()
    y = df_final.Missed_Seats_Pct.values
    #y.sort()
    
    ax.scatter(x, y, s=120, alpha=0.4, edgecolors='k', c='k')
    ax.axes.set_xlim(0,50)
    ax.axes.set_ylim(0,25)
    ax.set_xlabel('% Uncontested Seats', size=12)
    ax.set_ylabel('% Lost Seats', size=12)
   
    ax.set_title('% Seats Lost, % Uncontested Seats Correlation', size=14)
   
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    #plt.plot(x, y, 'o', label='original data')
    #plt.plot(x, intercept + slope*x, 'k', linewidth=0.05, label='fitted line')

    r2 = r_value**2
    r2_txt = 'R-Squared = ' + str(round(r2*100,2)) + '%'
    ax.annotate(r2_txt, xy=(32, 23), fontsize=12, fontweight='bold') #, xycoords='figure points')


if __name__ == "__main__":

     pass

