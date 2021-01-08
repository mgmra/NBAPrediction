import pandas as pd
import glob as glob
import time
from functools import reduce 
#%% wgranie danych, stworzenie tidy df do analiz zmiennych, oczyszczenie danych
pd.set_option('display.max_columns', 500)

files = glob.glob("Data/raw/*.csv")

#Wgranie i oczyszczenie danych, zduplikowane kolumny, indeksy
dfs = []
dfs_tidy = []
for file in files: 
    df = pd.read_csv(file, encoding="UTF-8")
    to_drop = list(df.filter(regex= 'Unnamed'))
    to_drop.append("Season")             
    df.drop(to_drop, axis=1, inplace=True)
    df.columns = df.columns.str.replace('\xa0', ' ')
    
    
    df = df.replace({'%': ''}, regex=True)
    df['Won'] = df['W/L'].replace({'W':True, 'L':False})
    df.drop(['W/L'], axis=1, inplace=True)
    df['Opponent'] = df['Match Up'].str[-3:]
    teams_sorted = ['_'.join(sorted(t)) + '_' for t in list(df[['Team', 'Opponent']].to_records(index=False))]
    df['Match_id'] = teams_sorted + df['Game Date']
    df['Match_Team_id'] = df['Team'] + "_" + df['Match_id']
    df.set_index('Match_Team_id', inplace=True)
    
    df_tidy = pd.melt(df, id_vars=['Team', 'Match_id', 'Game Date', 'Opponent', 'Won', 'Match Up', 'season'])        
    dfs.append(df)
    dfs_tidy.append(df_tidy)

#main_df_tidy i main_df wymagają zamiany kolumn zawierających procenty na wartoci bez procentów
main_df_tidy = pd.concat(dfs_tidy)

main_df = dfs[0].copy()
for i in range(1, len(dfs)):
    cols_to_use = dfs[i].columns.difference(main_df.columns)
    main_df = main_df.join(dfs[i][cols_to_use], how='inner')

#fix wrong datatypes    
main_df = main_df.astype({'OppOREB%':'float', 'OppeFG%':'float'})
main_df['Game Date'] = pd.to_datetime(main_df['Game Date'])


main_df['Game Number'] = main_df.groupby(['season', 'Team'])['Game Date'].rank()

#Fix single Data Error - Portland vs Memphis Match Up - Memphis was Home Team
main_df.loc['MEM_MEM_POR_12/08/2016', 'Match Up'] = main_df.loc['MEM_MEM_POR_12/08/2016', 'Match Up'].replace('@', 'vs.')

#Create Home/Away Indicator based on Match Up
main_df.loc[main_df['Match Up'].str.contains('vs.'), 'Home/Away'] = 'Home'
main_df.loc[main_df['Match Up'].str.contains('@ '), 'Home/Away'] = 'Away'


### Set Index Layer for Main DF
#Won and PTS will be used as target and as column to calculate %of won in last X matches
main_df['Won_Result'] = main_df['Won']
main_df['PTS_Result'] = main_df['PTS']
#index_list = ['Game Date', 'Game Number', 'Match_id', 'Match_Team_id', 'Home/Away', 'Won_Result', 'PTS_Result', 'Opponent']
index_list = ['Team', 'season', 'Game Date', 'Game Number', 'Match_id', 'Match_Team_id', 'Home/Away', 'Won_Result', 'PTS_Result', 'Opponent']
main_df.reset_index(inplace=True)
main_df.set_index(index_list, inplace=True)

#%% Df_team creation - rolling averages and subtraction


### Start measuring time of calculations
PRF_start_time = time.time()

#Calculate Rolling Means of metrics from last 3 matches - WARNING - TRY EMA INSTEAD

main_df = main_df.sort_index(level=[0, 1, 3]).drop(['Match Up'], axis=1)

#Rivarly index for convienient calculation of rivals history statistics
main_df['Rivalry'] = main_df.index.get_level_values('Match_id').str[:7]
main_df.set_index('Rivalry', append=True, inplace=True)




#Calculate 
def make_multi_columns(df, method_name): 
    head_index = [method_name]*len(df.columns)
    current_cols = df.columns
    df.columns = [head_index, current_cols]
    return df

#remove_vovels to create unified suffixes for df indices
def rem_vowel(string): 
    vowels = ('a', 'e', 'i', 'o', 'u')  
    for x in string.lower(): 
        if x in vowels: 
            string = string.replace(x, "")
    return string

#create_suffix composed of shortened multiindex names
def make_grp_suffix(df, grp_levels):
    grp_suffix = str()
    for g in grp_levels:
        index_name = str(df.index.names[g])
        if len(index_name)>4: 
            index_name = rem_vowel(index_name)[0:3]    
        suffix = "_" + index_name.lower()
        grp_suffix += suffix  
    return grp_suffix

    
def lag_ewm_df(df, alpha_param, grp_levels):
    ewm_df = df.groupby(level = grp_levels).transform(lambda x: x.ewm(alpha=alpha_param).mean())
    ewm_df = df.groupby(level = grp_levels).shift(1)
    
    name_suffix = make_grp_suffix(df,grp_levels)
    name_str = 'ewm_' + str(alpha_param) + str(name_suffix)
    
    ewm_df = make_multi_columns(ewm_df, name_str)
    return ewm_df
    
def lag_roll_df(df, window_size, grp_levels):

    df_roll = main_df.groupby(level = grp_levels, as_index=False).rolling(window_size).mean().reset_index(level=[0], drop=True)
    df_roll = df_roll.groupby(level = grp_levels, as_index=False).shift(1)

    name_suffix = make_grp_suffix(df, grp_levels)        
    name_str = 'roll_' + str(window_size) + str(name_suffix)
    
    df_roll = make_multi_columns(df_roll, name_str)
    return df_roll

alpha_list = [0.3, 0.5, 0.7]
grp_levels_ewm =  [[1, 0], [0]]
window_size_list = [2,3,5]
grp_levels_roll = [[1,0]]

ewm_dfs = [lag_ewm_df(main_df, a, g) for a in alpha_list for g in grp_levels_ewm]
roll_dfs = [lag_roll_df(main_df, ws, g) for ws in window_size_list for g in grp_levels_roll] 

df_team = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), ewm_dfs + roll_dfs)

#Last game result should be found based on rivalry - alphabetical concat of team names (already used for building IDs)
df_team.reset_index(level=['Game Date','Won_Result'], inplace=True)
df_team['Last_Game_Res'] = df_team.groupby(level=['Rivalry', 'Team'])['Won_Result'].shift(1)

#Calculation of Team's Won% in Season
df_team['Won_Result_NUM'] = df_team['Won_Result'].astype(int)
df_team['Number_Won_Matches'] = df_team.sort_index(level=['Game Number']).groupby(level=['Team', 'season'])['Won_Result_NUM'].cumsum().shift(1)
df_team['Won%'] = df_team['Number_Won_Matches']/df_team.reset_index(level=['Game Number'])['Game Number']                                                                                                                
df_team.drop(['Won_Result_NUM', 'Number_Won_Matches'], axis='columns', inplace=True)

#Calculation of teams ranking 
df_team['Ranking'] = df_team.sort_values(by=['Won%'], ascending=False).groupby(level=['season', 'Game Number'])['Won%'].rank(ascending=False, method = 'max')

#Setting Indices to avoid subtraction
df_team.set_index(['Last_Game_Res', 'Won_Result'], append=True, inplace=True)

#Calculate Rest_Days - number of days since teams last game - currently grouped by season, maybe should not
df_team['Rest_Days'] = df_team['Game Date'].subtract(df_team.groupby(level=['Team', 'season'])['Game Date'].shift(1)).dt.days
df_team.set_index('Game Date', append=True, inplace=True)

#Turn PTS_result to column so it can be subtracted between teams along with other vars
df_team.reset_index(level=['PTS_Result'], inplace=True)

#Subtract moving averages of home and away team 
df_team_home = df_team[df_team.index.get_level_values('Home/Away') == 'Home'].sort_index(level='Match_id')
df_team_away =  df_team[df_team.index.get_level_values('Home/Away') == 'Away'].sort_index(level='Match_id')

#Some cols are non sutractable, will be joined later on

#cols_non_subtract = ['Team', 'Game Date', 'Game Number', 'Match_Team_id', 'Home/Away', 'Opponent', 'Last_Game_Res', 'Last_Game_Date', 'season', 'Rivalry', 'Rest_Days']
#f_team_strct = df_team_home.drop(cols_non_subtract, axis=1).subtract(df_team_away.drop(cols_non_subtract, axis=1), axis=1)
#level_to_join = 'Match_id'
#levels_to_omit = [i for i,j in enumerate(df_team.index.names) if j != level_to_join]

df_team_strct  = df_team_home.subtract(df_team_away.values, axis=1)
df_team_strct.reset_index(level=['Last_Game_Res'],inplace=True)
#join selected non_subtractable cols

#df_team_strct = df_team_strct \
#                .join(df_team_home[['Team', 'Game Number', 'Rest_Days', 'Last_Game_Res', 'Last_Game_Date', 'Game Date', 'season', 'PTS_Result']].rename(columns=lambda c:  c+'_H' if c in ['Team','Game Number', 'Rest_Days', 'PTS_Result'] else c), on='Match_id') \
#                .join(df_team_away[['Team', 'Game Number', 'Rest_Days', "PTS_Result"]].rename(columns=lambda c: c+"_A"), on='Match_id')

#Calculate PTS Difference between Home and Away team for the target
#df_team_strct['PTS_Diff_Result'] = df_team_strct['PTS_Result_H'] - df_team_strct['PTS_Result_A'] 

#Convert Home Team Win - a target to Boolean
#df_team_strct['Won_Result'] = df_team_strct['Won_Result'].replace({-1:False,1:True}).astype('bool') 
df_team_strct = df_team_strct.apply(pd.to_numeric)
df_team_strct.info() 

print("--- %s seconds ---" % (time.time() - PRF_start_time))

#%%zapis pliku csv
#save file for further analysis, e.g. Tableau
df_team_strct.to_csv('Data/preprocessed/df_ml.csv')
df_team_strct.to_hdf('Data/preprocessed/df_ml_hdf.h5', key='match')

#DataFrame for Feature Selection and ML processing
#df_ml.to_csv('Output/df_ml.csv')
'''
### Creation of winner loser Dataframe
#Split DF to lost and won as there are two records currently for each match
main_df_lost = main_df[main_df['Won'] == "L"].set_index('Match_id') 
main_df_won = main_df[main_df['Won'] == "W"].set_index('Match_id')

# delete selected columns that are the same for won and lost team from one of data frames
duplicated_info_vars = ['Match Up', 'Game Date', 'MIN', 'season']
main_df_lost.drop(duplicated_info_vars, axis = 'columns', inplace = True)

#Dataframe to analyze winners and losers
df_winner_loser = main_df_won.join(main_df_lost, lsuffix = "_W", rsuffix = '_L')
'''
#UWAGA - MEMPHIS BYŁO GOSPODARZEM
#ids_h = list(df_team_home.index.get_level_values('Match_id'))
#ids_a =list(df_team_away.index.get_level_values('Match_id'))
#ah_diff = list(set(ids_a).difference(ids_h))
#diff_matches = main_df[main_df.index.get_level_values('Match_id').isin(ah_diff)]
