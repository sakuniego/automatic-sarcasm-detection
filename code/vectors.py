# import packages
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re # regular expressions
import warnings
import multiprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
    # TODO: Clean more
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

def similar_words(model, word):
    print("Similar words to", word + ":")
    print("(Similarity scores rounded to 4 decimals)")

    i = 0
    while (i < len(model.wv.most_similar(word))):
        print("Word:", model.wv.most_similar(word)[i][0], "- Similarity score:",
              round(model.wv.most_similar(word)[i][1],4))
        i += 1

# referenced csmoon-ml
def make_scatter():
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(1, 1, 1)

    pos_found_x = []
    pos_found_y = []
    found_names = []

    pos_rest_x = []
    pos_rest_y = []



def make_learn_vec(model, tag_data):
    vals = tag_data.values
    targets, regressors = zip(*[(post.tags[0], model.infer_vector(post.words, steps = 20)) for post in vals])
    return targets, regressors

def create_model(tag_train): # rename to create_d2v_model
    max_epochs = 10
    global model  # defining as global var so accessible outside of this func TODO: might be a bad idea
    model = Doc2Vec(workers = cores) # using dbow

    model.build_vocab(tag_train)
    model.train(tag_train, total_examples = model.corpus_count, epochs = max_epochs)

    model.save("d2v_sarc.model")

def create_mlp():
    # save results of d2v model in main
    # do train test split

    print("null")

def main():
    # 1. DATA INITIALIZATION
    in_csv = "train-balanced-sarcasm.csv" # csv containing data
    in_df = parse(in_csv)

    # raw_data_vis(in_df) # shows details about the raw data count
    # print_post(in_df, 0) # replace second parameter with integer within [0, 1010825]

    start_time_1 = time.time()
    print("STATUS UPDATE: Cleaning text...")

    in_df = in_df.fillna("na") # TODO: did to avoid TypeError, INSTEAD DELETE NULL VALUES IN PARSE AND REMOVE THIS
    in_df['comment'] = in_df['comment'].apply(clean_txt)
    in_df['parent_comment'] = in_df['parent_comment'].apply(clean_txt)

    # printing how long process took (in minutes rounded up)
    print("Took <%s minutes to clean text\n" % math.ceil((time.time() - start_time_1)/60))

    # 2. MODEL CREATION
    # Preparing data...
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

    # Model creation...
    """
    print("STATUS UPDATE: Creating and training model...")
    start_time_3 = time.time()
    create_model(tag_train)
    print("Took <%s minutes to create/train model\n" % math.ceil((time.time() - start_time_3)/60))
    """

    # 3. USING MODEL
    model = Doc2Vec.load("d2v_sarc.model") # loading existing model
    print("STATUS UPDATE: Loaded pre-trained model\n")

    # Find similar words...
    # word = "nevada" # change this to the word you'd like to see
    # similar_words(model, word)

    # Training classifier... (log reg)
    print("STATUS UPDATE: Training classifier...")
    start_time_4 = time.time()
    #TODO: put in own method
    train_y, train_X = make_learn_vec(model, tag_train)
    test_y, test_X = make_learn_vec(model, tag_test)

    classifier = LogisticRegression(n_jobs = 1, C = 1e5)
    classifier.fit(train_X, train_y)
    print("Took <%s minutes to train classifier\n" % math.ceil((time.time() - start_time_4) / 60))

    prediction = classifier.predict(test_X)
    #TODO: put logistic regression graph here

    print("Testing scores...")
    print("Accuracy: %s" % accuracy_score(test_y, prediction))
    print("f1 score: {}".format(f1_score(test_y, prediction, average = "weighted")))

main() # run main function