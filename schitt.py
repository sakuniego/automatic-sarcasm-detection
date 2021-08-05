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

def make_learn_vec(model, tag_data):
    vals = tag_data.values
    targets, regressors = zip(*[(post.tags[0], model.infer_vector(post.words, steps = 20)) for post in vals])
    return targets, regressors

def create_model(tag_train): # rename to create_d2v_model
    max_epochs = 10
    # global model  # defining as global var so accessible outside of this func TODO: might be a bad idea
    model = Doc2Vec(workers = cores) # using dbow

    model.build_vocab(tag_train)
    model.train(tag_train, total_examples = model.corpus_count, epochs = max_epochs)

    return model

def create_mlp(train_vecs, train_labels):
    # save results of d2v model in main
    # do train test split
    scale_vecs = StandardScaler()
    scaled_train_vecs = scale_vecs.fit_transform(train_vecs)
    # TODO: do above for test vecs too
    # scaled_test_vecs = scale_vecs.transform(test_vecs)

    # one hidden layer of size 100 TODO: Change hidden layers
    basic_clf = MLPClassifier(hidden_layer_sizes=(100,),activation="relu",random_state=1).fit(scaled_train_vecs,
                                                                                              train_labels)
    # label_pred = basic_clf.predict(scaled_test_vecs)

    # print(basic_clf.score(scaled_test_vecs, test_labels))

    # fig = plot_confusion_matrix(basic_clf, scaled_test_vecs, test_labels, display_labels = [0, 1])
    # fig.figure_.suptitle("Sarcasm Confusion Matrix")
    # plt.show()

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
    model = create_model(tag_train)
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

def main():
    # 1. DATA INITIALIZATION
    in_csv = "./data/season_1_plain.csv" # csv containing data
    in_df = parse(in_csv)
    new_df = in_df

    # raw_data_vis(in_df) # shows details about the raw data count
    # print_post(in_df, 0) # replace second parameter with integer within [0, 1010825]

    start_time_1 = time.time()
    print("STATUS UPDATE: Cleaning text...")

    in_df = in_df.fillna("na") # TODO: did to avoid TypeError, INSTEAD DELETE NULL VALUES IN PARSE AND REMOVE THIS
    in_df['line'] = in_df['line'].apply(clean_txt)

    # printing how long process took (in minutes rounded up)
    print("Took <%s minutes to clean text\n" % math.ceil((time.time() - start_time_1)/60))

    # 2. MODEL CREATION
    # Preparing data...
    start_time_2 = time.time()
    print("STATUS UPDATE: Tokenizing data...")

    # Splitting data into training and testing sets
    # train, test = train_test_split(in_df, test_size = 0.3, random_state = 42)
    train = in_df
    # test = test_df # TODO: replace with name of testing file

    # how can parent comment factor in?
    #TODO: should tags be unique? multiple tags?
    tag_train = train.apply(lambda r: TaggedDocument(words = tokenize(r['line']), tags = [r.tag]), axis = 1)
    #tag_test = test.apply(lambda r: TaggedDocument(words = tokenize(r['line']), tags = [r.tag]), axis = 1)

    print("Took <%s minutes to tokenize data\n" % math.ceil((time.time() - start_time_2)/60))

    # print(tag_train.values[0]) # TODO: Remove

    # Model creation... TODO: comment out
    ''' 
    print("STATUS UPDATE: Creating and training model...")
    start_time_3 = time.time()
    filename = "./models/d2v_schitt.model"
    model = create_model(tag_train)
    model.save(filename)
    print("Took <%s minutes to create/train model\n" % math.ceil((time.time() - start_time_3)/60))
    '''

    # 3. USING MODEL
    model = Doc2Vec.load("./models/d2v_schitt.model") # loading existing model #TODO: replace with filename
    print("STATUS UPDATE: Loaded pre-trained model\n")

    # Find similar words...
    #word = "motel" # change this to the word you'd like to see
    #similar_words(model, word)

    #print(len(model['TRAIN_0_1'])) #TODO:should remove

    # Training classifier... (log reg)
    print("STATUS UPDATE: Training classifier...")

# TODO: update 265, put this in own function
    lines = len(model.dv.index_to_key) # stores number of total utterances (lines)

    train_vecs = numpy.zeros((lines, 100)) # 265 lines represented by len 100 vector
    train_labels = numpy.zeros(lines) # 1 if sarcastic/rhetorical, 0 if not

    # TODO: update 265
    # TODO: remember range is non-inclusive
    # stores vectors and whether they're sarcastic or not in separate arrays
    for i in range (0,lines-1): # TODO: Why did this not work w/o -1?
        tag_not = "TRAIN_0_" + str(i)
        tag_sarc = "TRAIN_1_" + str(i)
        if tag_not in model.dv.index_to_key:
            train_vecs[i] = model[tag_not]
            train_labels[i] = 0
        else:
            train_vecs[i] = model[tag_sarc]
            train_labels[i] = 1

    # TODO: make this into its own method!
    # turning vectors into comma-separated string to create updated csv w/ vectors
    vecs = []

    # TODO: why is there a blank line at the end of result file?
    for i in range(lines):
        j = 0
        str_rep = ""
        while j < 100:
            str_rep += str(train_vecs[i, j])
            if (j < 99):
                str_rep += ","
            j = j + 1
        vecs.append(str_rep)

    new_df['vectors'] = vecs # creating new col containing vectors
    #print(new_df)
    #print(model['TRAIN_0_1'])

    new_df.to_csv("./data/train_and_vectors") # saving to file

    print("Creating t-SNE models...")
    # Printing t-SNE model
    make_tsne(train_vecs, train_labels)
    # make_speaker_tsne(train_vecs, train_labels, in_df)
    # make_3d_tsne(train_vecs, train_labels)

    # creating actual models
    # create_mlp(train_vecs, train_labels) # TODO: Store result and print return value
    individual_mlp("David", in_df)

main() # run main function