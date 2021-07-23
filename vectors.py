# import packages
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import warnings
import multiprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
import time
import math

# setup
cores = multiprocessing.cpu_count()
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

"""
Description: prints out comment at index i, its parent and whether it's labelled sarcastic or not
Parameters:
    df: data in pandas dataframe form
    i: index of desired comment
"""
def print_post(df, i):
    if (i < 0 or i > 1010825): # parameter out of range
        print("print_post() error: Integer out of bounds, please choose number within [0, 1010825]\n")
    else:
        post = df[df.index == i][['parent_comment','comment','label']].values[0]
        print("Parent Comment:", post[0])
        print("Highlighted Comment:", post[1])
        if(post[2] == 1):
            print("Label: SARCASTIC\n")
        else:
            print("Label: NOT SARCASTIC\n")

# file -> pandas df
def parse(in_file):
    # in_df = pd.DataFrame(columns = ['label', 'comment', 'parent_comment'])
    df = pd.read_csv(in_file)
    df = df[['label', 'comment', 'parent_comment']] # label, comment, parent_comment
    # df = df[pd.notnull(df['comment'])] # remove null comments, would require re-indexing maybe don't do

    return df

def raw_data_vis(in_df):
    # visualizing dataset
    count_sarc = in_df['label'].value_counts()

    # seeing how many of each value (0,1) there are
    print("# of non-sarcastic comments:", count_sarc.values[0])
    print("# of sarcastic comments:", count_sarc.values[1])

    if (count_sarc.values[0] == count_sarc.values[1]):
        print("The data is balanced.\n")
    else:
        print("The data is unbalanced.\n")

    # barplot of data
    plt.figure(figsize=(12, 4))
    sns.barplot(x = count_sarc.index, y = count_sarc.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Sarcasm?', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

def clean_txt(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'http\S+', r'<URL>', text) # replace URLs
    text = text.lower()
    return text

def tokenize(data):
    tokens = []

    for s in nltk.sent_tokenize(data):
        for w in nltk.word_tokenize(data):
            tokens.append(w.lower()) # maybe should not make lowercase

    return tokens;

def create_model(tag_train):
    max_epochs = 10
    global model # defining as global var so accessible outside of this func TODO: might be a bad idea
    model = Doc2Vec(workers = cores) # using dbow

    model.build_vocab(tag_train)
    model.train(tag_train, total_examples = model.corpus_count, epochs = max_epochs)

    model.save("d2v_sarc.model")


def main():
    in_csv = "train-balanced-sarcasm.csv" # csv containing data
    in_df = parse(in_csv)

    # raw_data_vis(in_df) # shows details about the raw data count
    # print_post(in_df, 0) # replace second parameter with integer within [0, 1010825]

    # TODO: clean up data using bs4
    start_time_1 = time.time()
    print("STATUS UPDATE: Cleaning text...")

    in_df = in_df.fillna("na") # TODO: did to avoid TypeError, INSTEAD DELETE NULL VALUES IN PARSE AND REMOVE THIS
    in_df['comment'] = in_df['comment'].apply(clean_txt)
    in_df['parent_comment'] = in_df['parent_comment'].apply(clean_txt)

    # printing how long process took (in minutes rounded up)
    print("Took <%s minutes to clean text\n" % math.ceil((time.time() - start_time_1)/60))

    start_time_2 = time.time()
    print("STATUS UPDATE: Tokenizing data...")

    # Splitting data into training and testing sets
    train, test = train_test_split(in_df, test_size = 0.3, random_state = 42)

    # how can parent comment factor in?
    #TODO: should tags be unique? multiple tags?
    tag_train = train.apply(lambda r: TaggedDocument(words = tokenize(r['comment']), tags = [r.label]), axis = 1)
    tag_test = test.apply(lambda r: TaggedDocument(words = tokenize(r['comment']), tags = [r.label]), axis = 1)

    print("Took <%s minutes to tokenize data\n" % math.ceil((time.time() - start_time_2)/60))

    # print(tag_train.values[0]) # TODO: Remove

    print("STATUS UPDATE: Creating and training model...")
    start_time_3 = time.time()
    create_model(tag_train)
    print("Took <%s minutes to create/train model\n" % math.ceil((time.time() - start_time_3)/60))

    # TODO: delete
    # print(model.wv.most_similar("republican"))


main() # run main function


# TODO: this is for building d2v model

# take in and process data (turn into tagged documents iterable format)
# documents = 0

# create model
# model = Doc2Vec(documents,
                #vector_size=5,
                #window=2,
                #min_count=1,
                #workers=4)


