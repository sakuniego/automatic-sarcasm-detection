# Automatic Sarcasm Detection
Basic classification model using novel data from the comedy show *Schitt's Creek* (seasons 1 and 2) to perform sarcasm detection. Implemented with gensim's Doc2Vec and scikit-learn's MLPClassifier.

## How To Run
1) Clone repo
2) Run `schitt.py`

## Data
* `season_1_plain.csv`: includes..., used as training data for MLPClassifier
* `season_1_excel.xlsx`: excel document containing additional organization of above data

## Function Descriptions

## Potential Errors
* May need to import nltk and do nltk.download(‘punkt’) (in python console)
* May need to run pip install python-Levenshtein to suppress error
  * If on Linux machine may get error: unable to execute 'x86_64-linux-gnu-gcc': No such file or directory
    * Run: `sudo apt-get install python3.x-dev`
  * x should be replaced with subversion of python (e.g. 8 for python 3.8)
  * Run `python --version` to find current version
* https://stackoverflow.com/questions/24398302/bs4-featurenotfound-couldnt-find-a-tree-builder-with-the-features-you-requeste
pip install lxml

