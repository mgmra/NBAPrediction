import pandas as pd
import glob as glob

pd.set_option('display.max_columns', 500)
#display options
#from tabulate import tabulate
#pdtabulate=lambda df:tabulate(df,headers='keys')

#comment
#home Team vs away team
#away team @ home team

files = glob.glob("Data\*.csv")


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
for i in range(1, len(dfs)-1):
    cols_to_use = dfs[i].columns.difference(main_df.columns)
    main_df = main_df.join(dfs[i][cols_to_use], how='inner')

#fix wrong datatypes    
main_df = main_df.astype({'OppOREB%':'float', 'OppeFG%':'float'})
main_df['Game Date'] = pd.to_datetime(main_df['Game Date'])
main_df['Game Number'] = main_df.groupby(['season', 'Team'])['Game Date'].rank()

main_df['Home Team'] = False
main_df['Home Team'] = main_df['Match Up'].str.contains('vs.')

### Creation of winner loser Dataframe
#Split DF to lost and won as there are two records currently for each match
main_df_lost = main_df[main_df['Won'] == "L"].set_index('Match_id') 
main_df_won = main_df[main_df['Won'] == "W"].set_index('Match_id')

# delete selected columns that are the same for won and lost team from one of data frames
duplicated_info_vars = ['Match Up', 'Game Date', 'MIN', 'season']
main_df_lost.drop(duplicated_info_vars, axis = 'columns', inplace = True)

#Dataframe to analyze winners and losers
df_winner_loser = main_df_won.join(main_df_lost, lsuffix = "_W", rsuffix = '_L')


### Creation of home/away merged DF
main_df_home = main_df[main_df['Home Team'] == True].set_index('Match_id') 
main_df_away = main_df[main_df['Home Team'] == False].set_index('Match_id')

main_df_away.drop(duplicated_info_vars, axis = 'columns', inplace = True)
df_home_away = main_df_home.join(main_df_away, lsuffix = "_H", rsuffix = '_A')



#calculation of moving statistics - wyciągnąć wszystkie kolumny nadające się do liczenia, kolumny takiej jak home/away,
#winner/loser oddzielić i póżniej dokleić
#Wymagana unifikacja typów danych
non_calc_cols = ['Team_A', 'Match Up','Opponent_H', 'Opponent_A', 'season', 'Team_H', 'Game Date']

#SORT BY GAME NUMBER
df_mov_avg = pd.DataFrame()
for col in df_home_away.columns.difference(non_calc_cols):
    #df_mov_avg[col] = df_home_away.drop(non_calculatable_cols, axis=1).groupby(['season', 'Team_H'])[col].rolling(3).mean().groupby(['season', 'Team_H']).shift(1)
    df_mov_avg[col] = df_home_away.sort_values('Game Number_A').groupby(['season', 'Team_H'])[col].rolling(3).mean().groupby(['season', 'Team_H']).shift(1)

#rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)

#TEST
#main_df_tidy.to_csv("Output/NBAPredictMainDFTidy.csv")

#test_df = dfs[0] 

