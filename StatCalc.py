import pandas as pd
import glob as glob

files = glob.glob("Data\*.csv")


dfs = []
dfs_tidy = []
for file in files: 
    df = pd.read_csv(file, encoding="UTF-8")
    to_drop = list(df.filter(regex= 'Unnamed'))
    to_drop.append("Season")             
    df.drop(to_drop, axis=1, inplace=True)
    df.columns = df.columns.str.replace('\xa0', ' ')
    df['Opponent'] = df['Match Up'].str[-3:]
    teams_sorted = ['_'.join(sorted(t)) + '_' for t in list(df[['Team', 'Opponent']].to_records(index=False))]
    df['Match_id'] = teams_sorted + df['Game Date']
    df['Match_Team_id'] = df['Team'] + "_" + df['Match_id']
    df.set_index('Match_Team_id', inplace=True)
    
    df_tidy = pd.melt(df, id_vars=['Team', 'Match_id', 'Game Date', 'Opponent', 'W/L', 'Match Up', 'season'])        
    dfs.append(df)
    dfs_tidy.append(df_tidy)

main_df_tidy = pd.concat(dfs_tidy)

merged_df = dfs[0].copy()
for i in range(1, len(dfs)-1):
    cols_to_use = dfs[i].columns.difference(merged_df.columns)
    merged_df = merged_df.join(dfs[i][cols_to_use], how='inner')
    
    
merged_df['Game Date'] = pd.to_datetime(merged_df['Game Date'])
merged_df['Game Number'] = merged_df.groupby(['season', 'Team'])['Game Date'].rank()



#TEST
#main_df_tidy.to_csv("Output/NBAPredictMainDFTidy.csv")

#test_df = dfs[0] 

#https://rpubs.com/tsumner/statistics
#test_df_win = test_df1[test_df1['W/L'] == 'W']
#test_df_lost = test_df1[test_df1['W/L'] == 'L']



#test_df_fold = test_df[test_df['W/L'] == 'W'].merge(test_df_lost, on = 'Match_id', suffixes=("_W","_L"))


#test_df_melt = pd.melt(test_df, id_vars=['Team', 'Match_id', 'Game Date', 'Opponent', 'W/L', 'Match Up', 'season'])
