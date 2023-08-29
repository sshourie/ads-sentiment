from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
import os 
import pandas as pd
import glob
from webdriver_manager.chrome import ChromeDriverManager # as there are different verions of chrome
# for debugging/testing use these
# from pprint import pprint
# from bs4 import BeautifulSoup

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def download_data(ad_lists, User ,Pass, url='https://login.ispot.tv/', consolidate_files =True, dir_ = r"C:", file_path = r"C:\xyz.csv" ):
    """ Go to website, enter credentials and automate download of data for input ads 

    Args:
        ad_lists(list): list of ad_ids as taken from ispot.tv
        User (str): username
        Pass (str): password
        url (str, optional): url to launch. Defaults to 'https://login.ispot.tv/'.
        consolidate_files (bool, optional): if to combine data to a single file. Defaults to True.
        dir_ (regexp, optional): directory only needed if consolidate_files=True. Defaults to r"C:".
        file_path (regexp, optional): file_path only needed if consolidate_files=True. Defaults to r"C:\xyz.csv".
    """

    

    options = webdriver.ChromeOptions() ;
    # change the directory where files are downloaded
    prefs = {"download.default_directory" : r"C:\Users\SS\Documents\New Folder"}; 
    options.add_experimental_option("prefs",prefs);
    # This will launch a chrome window from python
    driver = webdriver.Chrome(executable_path='./chromedriver',chrome_options=options);
    # driver = webdriver.Chrome(ChromeDriverManager().install())

    driver.get(url) # go to this website

    # enter credentials to login
    email_input = driver.find_element(By.XPATH, "//input[@class='clickable form-control']")
    email_input.send_keys(User)
    email_input.send_keys(Keys.ENTER)
    time.sleep(1)
    pwd_input = driver.find_element(By.XPATH, "//input[@class='form-control']")
    pwd_input.send_keys(Pass)
    pwd_input.send_keys(Keys.ENTER)
    
    # ispot website you can iterate through a lot of advertisements and download their survey results and data
    
    for idx, lst_i in enumerate(ad_lists):
        # print(idx)
        driver.find_element(By.XPATH, "//button[@class='ButtonSecondary--small resetAll']").click() # clear Ads
        time.sleep(10)
        
        for uuid in lst_i:
            # loops through all uuids and adds them
            search_bar = driver.find_element(By.ID, "quickSearchInput" )
            search_bar.send_keys(uuid)
            time.sleep(3)
            driver.find_element(By.XPATH, "//div[@class='listAd item']").click()
            time.sleep(1)
            driver.find_element(By.ID, "quickSearchInput" ).clear() 
        
        driver.find_element(By.XPATH, "//button[@class='ButtonPrimary']").click() # click on Done
        
        for element in driver.find_elements(By.XPATH, "//button[@title='Verbatim View']"): # click on all verbatim view
            element.click()
        driver.find_elements(By.XPATH, "//button[@title='Download Options']")[0].click() # click on download options
        
        driver.find_element(By.XPATH, "//button[@data-bind='click: exportAll']").click() # export all
    
    if consolidate_files:
        # save all the individual csv files into a single file
        all_files = glob.glob(dir_ + "/*.csv")
        qwe = []
        for filename in all_files:
            df = pd.read_csv(filename)
            qwe.append(df)
        df = pd.concat(qwe, axis=0)
        df.to_csv(file_path, index=False)


def sentiment_analysis(input_path, dest_path, colname = "Response"):
    """Sentiment analysis using pre-trained model from HuggingFace

    Args:
        input_path (str): file_path for csv containing the responses
        dest_path (_type_): file_path to save the sentiments
        colname (str, optional): Response column name. Defaults to "Response".
    """
    
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest" # can choose some other package from Huggingface instead
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # read the textual data from csv
    df = pd.read_csv(input_path)

    ret = []
    for i,text in enumerate(df[colname]):
        if not(i %1000): 
            print(i)
        encoded_input = tokenizer(str(text), return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ret.append({"Score":scores.max(), "Label": config.id2label[scores.argmax()]})
    out = pd.concat([df, pd.DataFrame(ret)], axis=1)

    out.to_csv(dest_path, index=False)
    
if __name__ == "main":
    pass
