#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Funkcja tworząca listę sezonów na podstawie rocznych dat graniczynych (YYYY - Y(Y+1))
#2.Pętla uruchamiająca Scraping na poszczególne sezony i sklejanie DFów, 
#dokładamy Season i sklejamy Data Frame'y
#3. Funkcja Scrapującą i Fetchująca surowe dane 
#4. Funkcja Czyszcząca i Zbierająca do DF


# In[2]:


from urllib.request import urlopen
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup as soup 
import time
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

#import org.openqa.selenium.JavascriptExecutor;
#import org.openqa.selenium.WebDriver;
#import org.openqa.selenium.chrome.ChromeDriver;
#import org.testng.annotations.Test;


# In[3]:


def scrape_NBA(URL, season):
    out = URL.format(season)
    return(out)

def list_seasons(start_year, end_year):
    seasons = [str(year) + "-" +str(year+1)[-2:] for year in range(start_year, end_year)]
    return seasons

def scrape_stat_table(URL, season, cookies=False):
    driver.get(URL.format(season))
    
    if cookies == True:
        time.sleep(5.0)
        accept_cookies = driver.find_element_by_xpath('//button[@id="onetrust-accept-btn-handler"]')
        accept_cookies.click()
        
    time.sleep(10.0)
        
    driver.execute_script("window.scrollTo(0, 100)") 
    
    time.sleep(10.0)
        
    select_all_pages = Select(driver.find_element_by_xpath('//select[@ng-options="p for p in vm.data.pages"]'))
    select_all_pages.select_by_index(0)
    
    time.sleep(2.0)    
    
    stat_table = driver.find_element_by_xpath('//div[@class="nba-stat-table"]')
    stat_table_html=stat_table.get_attribute('innerHTML')
    
    time.sleep(30.0)
    
    df=pd.read_html(stat_table_html)[0]
    df['season'] = season
    
    return df
    
    


# In[12]:


WebDriverPath=r"C:\Users\mateu\PycharmProjects\NBAPredict\WebDriver\chromedriver.exe"
#URL_trd ='https://stats.nba.com/teams/boxscores-traditional/?Season={}&SeasonType=Regular%20Season&dir=1'
URL_base ='https://stats.nba.com/teams/boxscores-{}/?Season={}&SeasonType=Regular%20Season&dir=1'
seasons = list_seasons(1996, 2020)
stat_types = ['scoring']




# In[13]:


dfs={}

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options, executable_path=WebDriverPath)


with driver as driver:
    URL = URL_base.format(stat_types[0], {})
    dfs[stat_types[0] + ": " + seasons[0]] = scrape_stat_table(URL, seasons[0], cookies=True)
    for stat_type in stat_types:
        URL = URL_base.format(stat_type, {})
        for season in seasons[1:]:
            dfs[stat_type + ": " + season] = scrape_stat_table(URL, season, cookies=False)
        

# In[16]:


##################TEST END
#ff_df = [dfs[key] for key in dfs.keys() if 'four-factors' in key]
#misc_df = [dfs[key] for key in dfs.keys() if 'misc' in key]
scr_df = [dfs[key] for key in dfs.keys() if 'scoring' in key]
        
#pd.concat(ff_df).to_csv("four-factors1.csv")
#pd.concat(misc_df).to_csv("misc1.csv")
pd.concat(scr_df).to_csv("scoring1.csv")


# In[ ]:




