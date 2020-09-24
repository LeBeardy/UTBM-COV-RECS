import requests,re
import pandas as pd
import bs4
from bs4 import BeautifulSoup
from PyPDF2 import PdfFileReader
import io

def get_data_from_pdf(link_to_pdf, category, title,author,link_to_article,date):
    """
    This function fetch the differents datas from the pdf linked to the article.
    It also format the data correctly.
    :param link_to_pdf:     Link to the pdf
    :param category   :     Category of the article
    :param title      :     Title of the article
    :param author     :     Author of the article
    :param link_to_article: Link to the article
    :param date       :     Date of publication of the article
    :return:                The formated data
    """
    formated_data = {}
    fetched_pdf = requests.get(link_to_pdf)
    pdf_content = io.BytesIO(fetched_pdf.content)
    pdf_reader = PdfFileReader(pdf_content)

    number_of_pages = pdf_reader.getNumPages()
    for x in range(0, number_of_pages):content=''.join(pdf_reader.getPage(x).extractText().split('\n'))

    formated_data = pd.DataFrame({'Category': [category], 'Title': [str(title)],'Author': [str(author)], 'Link': [str(link_to_article)], 'Number of pages': [number_of_pages], 'Content': [str(content)],'Date': [str(date)]})

    return formated_data


def extract_from_url():
    """
    This function extract the articles from the website: Groupe de veille covid.
    :return: the complete list of article extracted
    """
    merged = {}   # Request Page
    url_to_fetch = "https://www.groupedeveillecovid.fr/bibliographie-liste/"
    fetched_page = requests.get(url_to_fetch)
    page_content = fetched_page.text
    # find ".pdf" in a string
    match = re.compile('\.(pdf)')
    list_content = []

    # Instantiate BeautifulSoup
    soup_content = BeautifulSoup(page_content, features="lxml")
    category_links = soup_content.find_all("div", {"class": "tagcloud"})
    urls_to_scrape = []

    # Get the differents categories
    for tagcloud in category_links:
        for link in tagcloud.find_all('a', href=True):
            urls_to_scrape.append(link['href'])

    # Scrape all titles and abstracts
    for category_articles_url in urls_to_scrape:
        # Build the category name
        category_name = category_articles_url[:-1]
        # Removes the last trailing slash
        category_name = category_name.rsplit('/', 1)[1] # Get thelast part of the URL after a "/"

        fetched_category = requests.get(category_articles_url)
        category_content = fetched_category.text

        # Instantiate BeautifulSoup
        soup_category = BeautifulSoup(category_content, features="lxml")
        articles_list = soup_category.find_all("div", {"class": "btReadArticle"})

        for article_link_container in articles_list:
            for article_link in article_link_container.find_all('a', href=True):
                fetched_article = requests.get(article_link['href'])
                article_content = fetched_article.text
                # Instantiate BeautifulSoup
                soup_article = BeautifulSoup(article_content, features="lxml")

                title = soup_article.find("span", {"class": "headline"}).text

                author_element = soup_article.find(class_= "btText").find_all('p')

                authors_element = soup_article.find(class_= "btText").find_all('em')
                link_to_article = article_link['href']

                date = soup_article.find("span", {"class": "btArticleDate"}).text


                if not authors_element:
                    try:
                        for elem in author_element[1]:
                            if type(elem) is not bs4.element.Tag:
                                try:
                                    cont = elem.split(':')[1]
                                except IndexError:
                                    cont = 'null'
                                author = cont.lstrip().rstrip('.')
                            else:
                                author = 'null'
                    except IndexError:
                        author = 'null'
                else:
                    for elem in authors_element:
                        author = elem.text.lstrip().rstrip('.')


                for link in soup_article.findAll('a'):
                    try:
                        link_to_pdf = link['href']
                        if re.search(match, link_to_pdf):
                            data = get_data_from_pdf(link_to_pdf,category_name,title,author,link_to_article,date)
                            list_content.append(data)
                            merged =  pd.concat(list_content)
                            print(merged)
                    except KeyError:
                        pass

    return merged

def extract_articles_to_csv():
    """
    This function save the extracted data into a file .csv
    """
    articles_data = extract_from_url()

    articles_data.to_csv("article_data.csv", sep=';', encoding='utf-8-sig')

import dateparser

def format_date(data):
    """
    This function format the date of the data and add a column with the week and the month
    :param data: The data to format
    :return:     The formated data
    """
    data['Date'] = data['Date'].apply(dateparser.parse)
    data["SemMonth"] = data["Date"].dt.strftime("%a") + "_" + data["Date"].dt.strftime("%b")
    return data

def remove_duplicated(data):
    """
    This function remove the duplicate data in the dataset
    :param data: The data to format
    :return:     The formated data
    """
    data.sort_values('Title',inplace=True, ascending=False)
    duplicated_articles = data.duplicated('Title', keep = False)
    data = data[~duplicated_articles]
    return data

def remove_null_values(data):
    """
    This function remove the null data in the dataset
    :param data: The data to format
    :return:     The formated data
    """
    data = data.dropna()
    return data

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_stopwords(data):
    """
    This function remove the stopword into the article
    :param data: The data to format
    :return:     The formated data
    """
    stop_words = set(stopwords.words('english'))

    for title in data["Title"]:
        string = ""
        index = data[data['Title']==title].index.values
        for word in title.split():
            word = ("".join(e for e in word if e.isalnum()))
            word = word.lower()
            if not word in stop_words:
                string += word + " "
        data.at[index,"Title"] = string.strip()
    return data

def lemmatization(data):
    """
    This function lemmatize the article
    :param data: The data to format
    :return:     The formated data
    """
    lemmatizer = WordNetLemmatizer()
    for title in data["Title"]:
        string = ""
        index = data[data['Title']==title].index.values
        for word in word_tokenize(title):
            string += lemmatizer.lemmatize(word,pos = "v") + " "
        data.at[index, "Title"] = string.strip()
    return data

def get_data():
    """
    This function get the data from the csv and prepare it for the training
    :return:     The formated data
    """
    data=pd.read_csv('article_data.csv',sep=";")
    del data['Unnamed: 0']
    data = format_date(data)
    data = remove_duplicated(data)
    data = remove_null_values(data)
    data = remove_stopwords(data)
    data = lemmatization(data)
    return data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
import sklearn.metrics as metrics


def save_vectors(data):
    """
    This function create and save the vectors of word.
    :param data: The dataset to use
    :return:     The vectors
    """
    words_vectors = CountVectorizer()
    X_train_vectors = words_vectors.fit_transform((data.Content).values.astype('U'))
    #sauvegarde vecteur des mots
    pickle.dump(words_vectors.vocabulary_, open("words_vectors.pkl","wb"))
    return X_train_vectors

def tfidf_transformation(X_train_vectors):
    """
    This function transform the vectors into TF-IDF.
    :param X_train_vectors: The vectors to transform
    :return:                The TF-IDF
    """
    #transformation des vecteurs en TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_vectors)
    #sauvegarde TF-IDF
    pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))
    return X_train_tfidf

def svm_prediction(X_train_tfidf):
    """
    This function create and save the svm model for the predictions.
    It also test the model.
    :param X_train_tfidf: The TD-IDF to use for the svm model
    :return:              The predictions
    """
    svm_model = svm.LinearSVC()
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, data.Title, test_size=0.25, random_state=42)
    svm_model.fit(X_train_tfidf, data.Title)
    pickle.dump(svm_model, open("svm_model.pkl", "wb"))
    predicted_svm = svm_model.predict(X_test)
    result_svm = pd.DataFrame( {'true_titles': y_test,'predicted_titles': predicted_svm})
    print("accuracy %f" %metrics.accuracy_score(y_test, predicted_svm))

    return result_svm

def predict(X_train_tfidf, to_search):
    """
    This function predict the similar article with the keywords.
    :param X_train_tfidf: The TD-IDF to use for the svm model
    :param to_search:     Keywords to search 
    """
    svm_model = svm.LinearSVC()
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf,to_search, test_size=0.25, random_state=42)
    svm_model.fit(X_train_tfidf, data.Title)
    predicted_svm = svm_model.predict(X_test)
    result_svm = pd.DataFrame( {'titles': predicted_svm})
    return result_svm
#extract_articles_to_csv()
data = get_data()
vectors =save_vectors(data)
tfid_data=tfidf_transformation(vectors)
#training_predictions = svm_prediction(tfid_data)
#print(training_predictions.head())

arr = []
arr = ["diabetes  2019 covid" for i in range(len(data.Title))]


training_predictions_t = predict(tfid_data, arr)
print(training_predictions_t.head(6))

to_process = data["Title"].isin(training_predictions_t)
to_process = data[to_process]



import os
def verif_folders(folders):
    """
    This function test if the differentes folders exist, if not it create it
    :param folders: List of folders to test
    """
    if not os.path.isdir('./articles'):
        os.mkdir('./articles')

    for dir in folders:
        if not os.path.isdir('./articles/' + dir):
            os.mkdir('./articles/'+ dir)

def save_article(data):
    """
    This function save the articles in their category, in json format
    :param data: List of article to save
    """
    for index,article in data.iterrows():
        path = "./articles/%s/%s.json" % (article["Category"], article["Title"].replace(" ","_"))

        f = open(path, "w")
        f.write(article.to_json())
        f.close()

verif_folders(data.Category)
save_article(data.head(6))
