#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# **Seminar: Politische Debatten & Polarisierung im Bundestag**
## Sentimentanalyse im Kontext von Corona


# ## 1. Packages installieren

# In[1]:


# Import Basic Packages
import numpy as np                 # Numpy
import pandas as pd                 #Datafrane

# Import Visualization Packages
from collections import Counter     # um worte zu zählen
import matplotlib.pyplot as plt   # Für Visualisierung
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator #Wordcloud erstellen
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp


# Import NLP Packages
import nltk
import spacy



# ## 2. Methoden

# In[5]:


#  Methode 1: Gibt alle Reden aus dem DataFrame zurück, die den angegebenen Keywords (+Synonymen) entsprechen (nur und Konjunktion)

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def find_keywords_with_synonyms(text, keywords):
    found_keywords = set()
    words = word_tokenize(text)
    
    for word in words:
        if word in keywords:
            found_keywords.add(word)
            synonyms = get_synonyms(word)
            found_keywords.update(synonyms)
    
    return found_keywords

def filter_dataframe_by_keywords_with_synonyms(df, keywords):
    filtered_df = df[df['text'].apply(lambda x: bool(find_keywords_with_synonyms(x, keywords)))]
    return filtered_df


# In[6]:


#Methode 2:  Tokenisiert die Sätze in der Spalte 'text' des DataFrames und speichert sie in einem neuen DataFrame.

nltk.download('punkt')
from nltk.corpus import stopwords
  
# Load the appropriate language model

import spacy.cli 
spacy.cli.download("de_core_news_sm")
spacy.load('de_core_news_sm')


def tokenize_and_split_sentences(df):
    """
    Args:
    df (pandas.DataFrame): The DataFrame in which tokenization should be performed.

    Returns:
    pandas.DataFrame: The modified DataFrame with tokenized sentences.
    """
    # Tokenize sentences using NLTK
    df['tokenized_text'] = df['text'].apply(lambda x: nltk.sent_tokenize(x))

    # Create an empty DataFrame with the same columns
    columns = ['satz', 'id', 'period', 'date', 'name', 'party', 'redner_id', 'discussion_title']
    df_token_satz = pd.DataFrame(columns=columns)

    # Iterate over each row in the original DataFrame
    for _, row in df.iterrows():
        tokenized_text = row['tokenized_text']
        row_dict = row.to_dict()

        # Create a DataFrame with tokenized sentences
        sentences_df = pd.DataFrame({'satz': tokenized_text})

        # Merge the row data with the tokenized sentences DataFrame
        merged_df = pd.concat([sentences_df, pd.DataFrame([row_dict] * len(tokenized_text))], axis=1)

        # Append the merged DataFrame to the result DataFrame
        df_token_satz = pd.concat([df_token_satz, merged_df], ignore_index=True)

    return df_token_satz


# In[7]:


#Methode 3: zur Textbereinigung
import string
from nltk.corpus import stopwords
nlp = spacy.load('de_core_news_sm')

def clean_text(df, custom_stopwords=None):
    cleaned_sentences = []
    cleaned_tokens = []  # New list to store cleaned tokens
    
    # German stopwords
    stopwords_german = set(stopwords.words('german')) - {'nicht'} 
    
    # Update stopwords if custom stopwords are provided
    if custom_stopwords:
        stopwords_german.update(custom_stopwords)

    for sentence in df['satz']:
        # Tokenisierung mit Spacy
        doc = nlp(sentence)
        tokens = [token.text for token in doc if token.text not in string.punctuation]

        # Entfernung von Stoppwörtern mit NLTK
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_german]
        
        # Zusammenführen der bereinigten Tokens zu einem Satz
        cleaned_sentence = ' '.join(filtered_tokens)
        cleaned_sentences.append(cleaned_sentence)
        
        # Store cleaned tokens separately
        cleaned_tokens.append(filtered_tokens)

    # Assign the cleaned tokens to the DataFrame
    df['tokens'] = cleaned_tokens
    
    # Erstellung einer neuen Spalte 'cleaned_text' im DataFrame mit den bereinigten Sätzen
    df['cleaned_text'] = cleaned_sentences

    # Delete rows with empty 'cleaned_tokens'
    df = df[df['tokens'].map(len) > 0]
    
    return df



# In[8]:
#Methode 4: Zur Darstellung von nGrammen

from collections import Counter

def plot_most_frequent_ngrams(df, num_most_common=10):
    # Get the tokens from the DataFrame
    tokens = list(df['cleaned_text'].values)

    # Count unigrams
    unigram_counts = Counter()
    for text in tokens:
        unigrams = text.split()
        unigram_counts.update(unigrams)

    # Count bigrams
    bigram_counts = Counter()
    for text in tokens:
        unigrams = text.split()
        bigrams = [",".join(bigram) for bigram in zip(unigrams[:-1], unigrams[1:])]
        bigram_counts.update(bigrams)

    # Count trigrams
    trigram_counts = Counter()
    for text in tokens:
        unigrams = text.split()
        trigrams = [",".join(trigram) for trigram in zip(unigrams[:-2], unigrams[1:-1], unigrams[2:])]
        trigram_counts.update(trigrams)

    # Get the most frequent tokens
    most_common_unigrams = unigram_counts.most_common(num_most_common)
    most_common_bigrams = bigram_counts.most_common(num_most_common)
    most_common_trigrams = trigram_counts.most_common(num_most_common)

    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Plot most frequent unigrams
    axes[0].barh([str(gram) for gram, count in most_common_unigrams], [count for gram, count in most_common_unigrams])
    axes[0].set_title('Most Frequent Unigrams')

    # Plot most frequent bigrams
    axes[1].barh([str(gram) for gram, count in most_common_bigrams], [count for gram, count in most_common_bigrams])
    axes[1].set_title('Most Frequent Bigrams')

    # Plot most frequent trigrams
    axes[2].barh([str(gram) for gram, count in most_common_trigrams], [count for gram, count in most_common_trigrams])
    axes[2].set_title('Most Frequent Trigrams')

    plt.tight_layout()
    plt.show()





# In[9]:


# Methode 5 zur Modellentwicklung

from transformers import pipeline

def sentiment_analysis(df, text_column):
    # Define the sentiment analysis model
    nlp_sentiment = pipeline("sentiment-analysis", model='oliverguhr/german-sentiment-bert')

    # Apply sentiment analysis to the specified text column in the DataFrame
    df['Sentiment'] = df[text_column].apply(lambda x: nlp_sentiment(x))

    # Extract sentiment label and score
    df['Sentiment_Label'] = [x[0]['label'] for x in df['Sentiment']]
    df['Sentiment_Score'] = [x[0]['score'] for x in df['Sentiment']]

    # Remove the 'Sentiment' column
    df = df.drop(columns=['Sentiment'])

    return df



# In[13]:


#Methoden 6 zur Visualisierung des Sentiments

import plotly.express as px

def plot_sentiment_analysis(df_grundrechte_original, df_grundrechte_cleaned):
    # Count the frequency of each sentiment label
    df1_count = df_grundrechte_original['Sentiment_Label'].value_counts()
    df2_count = df_grundrechte_cleaned['Sentiment_Label'].value_counts()

    # Set the color palette
    colors = {'Positive': 'mediumseagreen', 'Negative': 'crimson', 'Neutral': 'royalblue'}

    # Create bar plots for sentiment distribution
    figure1 = px.bar(x=df1_count.index, y=df1_count.values, color=df1_count.index, color_discrete_map=colors)
    figure2 = px.bar(x=df2_count.index, y=df2_count.values, color=df2_count.index, color_discrete_map=colors)

    # Customize labels and titles
    figure1.update_layout(
        title_text='Sentiment Distribution - Original Text',
        title_font_size=24,
        xaxis_title='Sentiment',
        yaxis_title='Count',
        width=800,
        height=600
    )

    figure2.update_layout(
        title_text='Sentiment Distribution - Cleaned Text',
        title_font_size=24,
        xaxis_title='Sentiment',
        yaxis_title='Count',
        width=800,
        height=600
    )

    # Display the plots
    figure1.show()
    figure2.show()


# In[15]:


# Methoden 7 zur Visualisierung nach Parteizugehörigkeit

import plotly.graph_objects as go

def plot_sentiment_by_party (df):
    # Group the data by party and sentiment label and count the occurrences
    party_sentiment = df.groupby(['party', 'Sentiment_Label']).size().reset_index(name='Count')

    # Calculate the total count for each party
    party_count = party_sentiment.groupby('party')['Count'].sum()

    # Calculate the percentage of each sentiment category for each party
    party_sentiment['Percentage'] = party_sentiment.apply(lambda row: row['Count'] / party_count[row['party']] * 100, axis=1)

    # Create separate dataframes for each sentiment label
    positive_df = party_sentiment[party_sentiment['Sentiment_Label'] == 'positive']
    negative_df = party_sentiment[party_sentiment['Sentiment_Label'] == 'negative']
    neutral_df = party_sentiment[party_sentiment['Sentiment_Label'] == 'neutral']

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=positive_df['party'], y=positive_df['Count'], name='Positive', marker_color='mediumseagreen',
                         text=positive_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
                         textposition='auto'))
    fig.add_trace(go.Bar(x=negative_df['party'], y=negative_df['Count'], name='Negative', marker_color='crimson',
                         text=negative_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
                         textposition='auto'))
    fig.add_trace(go.Bar(x=neutral_df['party'], y=neutral_df['Count'], name='Neutral', marker_color='royalblue',
                         text=neutral_df['Percentage'].apply(lambda x: f'{x:.2f}%'),
                         textposition='auto'))

    fig.update_layout(
        barmode='group',
        xaxis_title='Partei',
        yaxis_title='Anzahl an Sätzen',
        title='Sentiment-Verteilung nach Parteizugehörigkeit'
    )

    fig.show()


# In[19]:


# Methode 9 zum Visualisierung von Wordclouds nach Sentiment
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_sentiment_wordclouds(df):
    # Group the data by sentiment label
    sentiment_groups = df.groupby('Sentiment_Label')
    text_by_sentiment = {}

    # Combine the text for each sentiment label
    for sentiment, group in sentiment_groups:
        text_by_sentiment[sentiment] = ' '.join(group['cleaned_text'].tolist())

    # Generate a word cloud for each sentiment
    for sentiment, text in text_by_sentiment.items():
        wordcloud = WordCloud(background_color='black', width=400, height=300, max_words=150, colormap='tab20c').generate(text)

        # Plot the word cloud
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(sentiment + ' Sentiment Word Cloud')
        plt.show()


