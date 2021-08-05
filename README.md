# Automatic Sarcasm Detection
Basic classification model using novel data from the comedy show *Schitt's Creek* (seasons 1 and 2) to perform sarcasm detection. Implemented with gensim's Doc2Vec and scikit-learn's MLPClassifier.

## How To Run
1) Clone repo
2) Install necessary packages in `schitt.py`
3) Run `schitt.py`

**Recommended IDE**: PyCharm

## Data
* `/data/season_#_plain.csv`: includes..., used as training data for MLPClassifier
* `/data/season_#_excel.xlsx`: excel document containing additional organization of above data
* `/schitts_scripts/`: contains unedited, edited (.docx) and .txt versions of schitt's creek scripts separated by seasons (all scripts sourced from [Forever Dreaming](https://transcripts.foreverdreaming.org/viewforum.php?f=132))
  * unedited: marked with speaker and labelled with sarcastic/rhetorical labels and descriptions
  * edited: paranthesized actions removed, colon delimiters inserted
  * txt: converted for easy import into excel (see below instructions)

### Importing .txt into Excel
1) Open Excel and select Open > Browse
2) Change file type from "All Excel Files" to "Text Files" and open desired text file
3) In the Text Import Wizard, under Original dataype, select "Delimited" radio button then hit "Next"
4) Under Delimiters unselect "Tab" checkbox, and next to "Other" type `:` (colon) then hit "Next"
5) Hit "Finish"

## Function Descriptions

## Visualizations

## Potential Errors
* May need to `import nltk` and do `nltk.download(‘punkt’)` (in Python Console)
* May need to run `pip install python-Levenshtein` (in Terminal) to suppress error
  * If on Linux machine may get error: unable to execute 'x86_64-linux-gnu-gcc': No such file or directory
    * Run: `sudo apt-get install python3.x-dev` (in Terminal)
      * x should be replaced with subversion of python (e.g. 8 for python 3.8)
      * Run `python --version` to find current version
* bs4.FeatureNotFound: Couldn't find a tree builder with the features you requested: lxml...
  * Run: `pip install lxml` (in Terminal)

### Credits
* This research project was funded by the National Science Foundation (IIS-1757929)
* Sakuni Egodawatte (Undergraduate Researcher, UW-Madison)
* Emily Hand (Faculty Advisor, UNR)
* David Feil-Seifer (REU Coordinator, UNR)
* Derek D. Stratton (Grad Advisor, UNR)

### Glossary
**Doc2Vec**: method for turning chunks of text (whole documents or even sentences) into vectors that can be more easily processed by a machine learning model

**sarcasm detection**: type of sentiment analysis, detecting sarcasm in given text

**sentiment analysis**: determining the emotion (and other subjective features) of a given text

#### Referenced Resources

#### Contact
If you have any questions, please don't hesitate to reach out.

Email: egodawatte@wisc.edu
LinkedIn: linkedin.com/in/sakuni
