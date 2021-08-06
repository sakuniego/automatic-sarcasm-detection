"""
Filename: schitt.py
Description:
Author: Sakuni Egodawatte
Last Updated: 08/05/2021

"""

# import packages
import matplotlib
import nltk
import numpy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from bs4 import BeautifulSoup
import re # regular expressions
import warnings
import multiprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time
import math

# setup
cores = multiprocessing.cpu_count()
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
plt.rcParams.update({'font.size': 22}) # make font-size of plot axes bigger

"""
Description: prints out comment at index i, its parent and whether it's labelled sarcastic or not
Parameters:
    df: data in pandas dataframe form
    i: index of desired comment
"""
# TODO: Fix
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
    df = pd.read_csv(in_file, encoding='cp1252')
    df = df[['tag', 'speaker', 'line', 'label']] # label, comment, parent_comment
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
    # TODO: Clean more (e.g. get rid of extra space before/after things)
    text = BeautifulSoup(text, "lxml").text
    # getting rid of punctuation
    text = re.sub(r'\.',r'',text)
    text = re.sub(r'\,', r'', text)
    text = re.sub(r'\?', r'', text)
    text = re.sub(r'\!', r'', text)
    text = re.sub(r'\.\.\.', r'', text)
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

# referenced jeffd23 from kaggle here
'''

'''
def make_tsne(vectors, labels):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    fitted_vecs = tsne_model.fit(vectors)
    new_vecs = fitted_vecs.embedding_ # get the 2d embeddings in vector form

    x = []
    y = []
    colors = ['darkgrey', 'red']

    for i in range(len(new_vecs)):
        x.append(new_vecs[i,0])
        y.append(new_vecs[i, 1])

    plt.figure(figsize=(16, 16))
    # s = 100 means size 10 points
    #plt.scatter(x, y, c = labels, cmap = matplotlib.colors.ListedColormap(colors), s = 100)
    for i in range(len(x)):
        if labels[i] == 0:
            plt.scatter(x[i], y[i], c='darkgrey', s=100)
    for i in range(len(x)):
        if labels[i] == 1:
            plt.scatter(x[i], y[i], c='red', s=100)
    plt.savefig('./visualizations/sarcasm_tsne.png', format = "png", transparent = True)

def make_speaker_tsne(vectors, labels, df):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    fitted_vecs = tsne_model.fit(vectors)
    new_vecs = fitted_vecs.embedding_ # get the 2d embeddings in vector form

    x = []
    y = []
    speaker_list = []
    speaker1 = "David"  # edit with a different speaker
    speaker2 = "Alexis"  # edit with a different speaker
    speaker3 = "Johnny"  # edit with a different speaker
    speaker4 = "Moira"  # edit with a different speaker
    #colors = ['grey', 'red', 'blue']

    for i in range(len(new_vecs)):
        x.append(new_vecs[i,0])
        y.append(new_vecs[i,1])

    for index, row in df.iterrows():
        speaker_list.append(row['speaker'])

    # converts speaker list to list of when selected speaker uses sarcasm
    for i in range(len(speaker_list)):
        if speaker_list[i] == speaker1 and labels[i] == 1:
            speaker_list[i] = 1
        elif speaker_list[i] == speaker2 and labels[i] == 1:
            speaker_list[i] = 2
        elif speaker_list[i] == speaker3 and labels[i] == 1:
            speaker_list[i] = 3
        elif speaker_list[i] == speaker4 and labels[i] == 1:
            speaker_list[i] = 4
        else:
            speaker_list[i] = 0

    plt.figure(figsize=(16, 16))
    # plt.scatter(x, y, c = speaker_list, cmap = matplotlib.colors.ListedColormap(colors), s=100)
    # for-loops ensure highlighted data is always on top
    for i in range(len(x)):
        if speaker_list[i] == 0:
            plt.scatter(x[i], y[i], c='darkgrey', s=100)
    for i in range(len(x)):
        if speaker_list[i] == 1:
            plt.scatter(x[i], y[i], c='red', s=100)
    for i in range(len(x)):
        if speaker_list[i] == 2:
            plt.scatter(x[i], y[i], c='blue', s=100)
    for i in range(len(x)):
        if speaker_list[i] == 3:
            plt.scatter(x[i], y[i], c='black', s=100)
    for i in range(len(x)):
        if speaker_list[i] == 4:
            plt.scatter(x[i], y[i], c='magenta', s=100)
    filename = "./visualizations/compare_sarcasm_tsne.png"
    plt.savefig(filename, format = "png", transparent = True)

def make_3d_tsne(vectors, labels):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault) # resetting stylesheet changes
    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
    fitted_vecs = tsne_model.fit(vectors)
    new_vecs = fitted_vecs.embedding_ # get the 3d embeddings in vector form

    fig = plt.figure(figsize=(16, 16))
    ax = plt.axes(projection='3d')

    x_0 = []
    y_0 = []
    z_0 = []

    x_1 = []
    y_1 = []
    z_1 = []

    colors = ['grey', 'red']

    for i in range(len(new_vecs)):
        x_0.append(new_vecs[i, 0]) # row i, col 0
        y_0.append(new_vecs[i, 1])
        z_0.append(new_vecs[i, 2])

    # TODO: plot separately and make red size bigger
    ax.scatter3D(x_0, y_0, z_0, c = labels, cmap = matplotlib.colors.ListedColormap(colors), s = 20)
    # ax.view_init(120, 30) # change view angle of plot
    filename = "./visualizations/sarcasm_3d_tsne.png"
    plt.savefig(filename, format = "png")

# referenced TDS
def make_learn_vec(model, tag_data):
    vals = tag_data.values
    targets, regressors = zip(*[(line.tags[0], model.infer_vector(line.words, steps = 20)) for line in vals])
    return targets, regressors

def create_d2v_model(tag_train): # rename to create_d2v_model
    max_epochs = 10
    # global model  # defining as global var so accessible outside of this func TODO: might be a bad idea
    model = Doc2Vec(workers = cores) # using dbow

    model.build_vocab(tag_train)
    model.train(tag_train, total_examples = model.corpus_count, epochs = max_epochs)

    return model

def create_mlp(train_vecs, train_labels, test_vecs, test_labels):
    # save results of d2v model in main
    scale_vecs = StandardScaler()
    scaled_train_vecs = scale_vecs.fit_transform(train_vecs)
    # TODO: do above for test vecs too
    scaled_test_vecs = scale_vecs.transform(test_vecs)

    # one hidden layer of size 100 TODO: Change hidden layers
    basic_clf = MLPClassifier(hidden_layer_sizes=(100,),activation="relu",random_state=1).fit(scaled_train_vecs,
                                                                                              train_labels)
    label_pred = basic_clf.predict(scaled_test_vecs)

    print(basic_clf.score(scaled_test_vecs, test_labels))

    matplotlib.rcParams.update(matplotlib.rcParamsDefault) # reset font-size
    fig = plot_confusion_matrix(basic_clf, scaled_test_vecs, test_labels) # TODO: set display labels
    fig.figure_.suptitle("Sarcasm Confusion Matrix")
    filename = "./visualizations/sarcasm_confusion_matrix.png"
    plt.savefig(filename, format="png", bbox_inches = "tight")

def individual_mlp(speaker, df_train): # TODO: Add df_test
    df_train = df_train[df_train['speaker'].str.contains(speaker)] # removing all rows where speaker col !contain speaker param
    df_train.reset_index(drop=True, inplace=True) # resetting indices
    # print(df_train)
    # print(df_train.loc[[1]]) TODO: remove
    # df_test = df_test[df_test['speaker'].str.contains(speaker)]
    # df_test.reset_index(drop=True, inplace=True)

    tag_train = df_train.apply(lambda r: TaggedDocument(words=tokenize(r['line']), tags=[r.tag]), axis=1)
    # tag_test = df_test.apply(lambda r: TaggedDocument(words = tokenize(r['line']), tags = [r.tag]), axis = 1)

    filename = "./models/d2v_" + speaker.lower() + ".model"
    model = create_d2v_model(tag_train)
    model.save(filename)

    model = Doc2Vec.load(filename)

    lines = len(model.dv.index_to_key)  # stores number of total utterances (lines)

    train_vecs = numpy.zeros((lines, 100))  # 265 lines represented by len 100 vector
    train_labels = numpy.zeros(lines)  # 1 if sarcastic/rhetorical, 0 if not

    for i in range (0,lines-1): # TODO: Why did this not work w/o -1?
        tag_not = "TRAIN_0_" + str(i)
        tag_sarc = "TRAIN_1_" + str(i)
        if tag_not in model.dv.index_to_key:
            train_vecs[i] = model[tag_not]
            train_labels[i] = 0
        elif tag_sarc in model.dv.index_to_key:
            train_vecs[i] = model[tag_sarc]
            train_labels[i] = 1

    # create_mlp(train_vecs, train_labels)

"""
Description: initializes and runs functions for Schitt's Creek data initialization, visualization and analysis and
records to time taken for each section
"""
def main():
    # 1. DATA INITIALIZATION
    # Desc: Importing, visualizing, cleaning data, storing in dataframes and tokenizing
    train_csv = "./data/season_1_plain.csv" # csv containing test data
    test_csv = "./data/season_2_plain.csv" # csv containing train data
    train_df = parse(train_csv)
    test_df = parse(test_csv)
    # used for creating new csv files w/ d2v vectors:
    train_vecs_df = train_df
    test_vecs_df = test_df

    # raw_data_vis(train_df) # shows details about the raw data count
    # print_post(train_df, 0) # replace second parameter with integer within [0, 1010825]

    start_time_1 = time.time()  # used to calculate length of time to perform below task
    print("STATUS UPDATE: Cleaning data...\n")

    # done to avoid TypeError, fills all blank values w/ "na" (deleting null vals would delete valid data)
    train_df = train_df.fillna("na")
    test_df = test_df.fillna("na")
    train_df['line'] = train_df['line'].apply(clean_txt) # clean text
    test_df['line'] = test_df['line'].apply(clean_txt)

    # how long above task took (in minutes rounded up)
    print("Took <%s minutes to clean data\n" % math.ceil((time.time() - start_time_1)/60))

    start_time_2 = time.time()
    print("STATUS UPDATE: Tokenizing data...\n")

    # tagging script line with tag col
    # TODO: could just not do it separately and combine before and tag at same time
    # TODO: remove
    '''
    frames = [train_df, test_df]
    combined_df = pd.concat(frames)
    tag_all = combined_df.apply(lambda r: TaggedDocument(words=tokenize(r['line']), tags=[r.tag]), axis=1)
    '''
    tag_train = train_df.apply(lambda r: TaggedDocument(words=tokenize(r['line']), tags=[r.tag]), axis=1)
    tag_test = test_df.apply(lambda r: TaggedDocument(words=tokenize(r['line']), tags=[r.tag]), axis=1)

    print("Took <%s minutes to tokenize data\n" % math.ceil((time.time() - start_time_2) / 60))

    # print(tag_train.values[0]) # Uncomment to see what tagged data looks like

    # 2. D2V MODEL CREATION
    # Desc: Creating Doc2Vec model and saving
    filename = "./models/d2v_schitt.model"

    # TODO COMMENT OUT
    # TODO: Uncomment to update model
    print("STATUS UPDATE: Creating and training d2v model...\n")
    start_time_3 = time.time()
    
    model = create_d2v_model(tag_train)
    model.save(filename)
    
    print("Took <%s minutes to create/train d2v model\n" % math.ceil((time.time() - start_time_3)/60))

    # 3. USING MODEL
    # model = Doc2Vec.load(filename) # loading existing model TODO: UNCOMMENT
    print("STATUS UPDATE: Loaded pre-existing model\n")

    # Find similar words...
    #word = "motel" # change this to the word you'd like to see
    #similar_words(model, word)

    y_train, X_train = make_learn_vec(model, tag_train)
    y_test, X_test = make_learn_vec(model, tag_test)

    #print(y_test)
    y_train = list(y_train)
    y_test = list(y_test)

    for i in range(len(y_train)):
        tag_not = "TRAIN_0_" + str(i)
        tag_sarc = "TRAIN_1_" + str(i)
        if y_train[i] == tag_not:
            y_train[i] = '0'
        elif y_train[i] == tag_sarc:
            y_train[i] = '1'

    for i in range(len(y_test)):
        tag_not = "TEST_0_" + str(i)
        tag_sarc = "TEST_1_" + str(i)
        if y_test[i] == tag_not:
            y_test[i] = '0'
        elif y_test[i] == tag_sarc:
            y_test[i] = '1'

    y_train = tuple(y_train)
    y_test = tuple(y_test)

# TODO: update 265, put this in own function
    '''
    lines = len(model.dv.index_to_key) # stores number of total utterances (lines)
    lines_train = len(train_df) # num training utterances
    lines_test = len(test_df) # num testing utterances

    train_vecs = numpy.zeros((lines_train, 100)) # each line represented by len 100 vector
    train_labels = numpy.zeros(lines_train) # 1 if sarcastic/rhetorical, 0 if not
    test_vecs = numpy.zeros((lines_test, 100))
    test_labels = numpy.zeros(lines_test)

    # TODO: remember range is non-inclusive
    # put all this and below part int oone function
    # stores vectors and whether they're sarcastic or not in separate arrays
    for i in range (0, lines):
        tag_not_train = "TRAIN_0_" + str(i)
        tag_sarc_train = "TRAIN_1_" + str(i)
        tag_not_test = "TEST_0_" + str(i)
        tag_sarc_test = "TEST_1_" + str(i)
        if tag_not_train in model.dv.index_to_key:
            train_vecs[i] = model[tag_not_train]
            train_labels[i] = 0
        elif tag_sarc_train in model.dv.index_to_key:
            train_vecs[i] = model[tag_sarc_train]
            train_labels[i] = 1
        elif tag_not_test in model.dv.index_to_key:
            test_vecs[i] = model[tag_not_test]
            test_labels[i] = 0
        elif tag_sarc_test in model.dv.index_to_key:
            test_vecs[i] = model[tag_sarc_test]
            test_labels[i] = 1

    # train_vecs, train_labels = extract_vecs(model)

    # TODO: make this into its own method!
    # turning vectors into comma-separated string to create updated csv w/ vectors
    vecs = []

    # TODO: why is there a blank line at the end of result file?
    for i in range(lines_train):
        j = 0
        str_rep = ""
        while j < 100:
            str_rep += str(train_vecs[i, j])
            if (j < 99):
                str_rep += ","
            j = j + 1
        vecs.append(str_rep)

    train_vecs_df['vectors'] = vecs # creating new col containing vectors
    #print(train_vecs_df)
    #print(model['TRAIN_0_1'])

    train_vecs_df.to_csv("./data/train_and_vectors") # saving to file
    '''

    # print("Creating t-SNE models...")
    # Printing t-SNE model
    # make_tsne(train_vecs, train_labels) #TODO FIX TO USE RIGHT ARGS
    # make_speaker_tsne(train_vecs, train_labels, train_df)
    # make_3d_tsne(train_vecs, train_labels)

    print("STATUS UPDATE: Creating and training MLPClassifier model...\n")
    start_time_4 = time.time()

    # creating actual models
    create_mlp(X_train, y_train, X_test, y_test) # TODO: Store result and print return value
    # individual_mlp("David", train_df)

    # TODO REMOVE logreg model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print(logreg.score(X_test, y_test))

    print("Took <%s minutes to create/train MLPClassifier model\n" % math.ceil((time.time() - start_time_4) / 60))

main() # run main function