# UTBM-COV-RECS


This project is a LDA-based covid article recommender system for the UTBM (Université de technologie de Belfort Montbéliard) a french university.


## The crawler

To fetch the differents data about the articles, we use request and BeautifulSoup.

You have to use extract_articles_to_csv() to fetch the article and tosave it into a .csv

## Get the data

you have to use get_data() to get the data cleaned and ready to be processed

## Recommendations

The recommendation system use an SVM model to predict the correct article.
You have to use def predict(X_train_tfidf, to_search) with X_train_tfidf, the tf-idf model and to_search the keywords.

## Save the recommendations

To save the article into the correspondant category folder you have to use verif_folders(folders) to test if the arborescence is correct
then save_article(data) to save the articles.
## Usage
to install the required library
```bash
pip install requirements.txt
```

To test
  ```bash
  python ./UTBM-COV-RECS.py
  ```
Then you can use the differents endpoints find here : http://localhost:5000/api/ui/
