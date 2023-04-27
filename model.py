from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

df = pd.read_csv("dataset/Reddit_Data.csv",encoding = 'ISO-8859-1')

# Load Text Cleaning Pkgs
# import neattext.functions as nfx

# # Data Cleaning
# # User handles
# df['Clean_Text'] = df['Caption'].apply(nfx.remove_userhandles)

# # Stopwords
# df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

def train():

    Xfeatures = df['Clean_Text'].fillna(' ')
    ylabels = df['LABEL']

    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3)

    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression(max_iter=500))])
    pipe_lr.fit(x_train,y_train)

    print(f"Model trained with accuracy of {pipe_lr.score(x_test,y_test)}")
    return pipe_lr

# save model

def predcit(value):
    try:
        with open("model.pickle","rb") as f:
            print("Using pre trained model...")
    except :
        print("Training model...")
        pickle.dump(train(), open("model.pickle","wb"))
    model = pickle.load(open("model.pickle","rb"))
    result = model.predict([value])
    return result[0]

if __name__ == "__main__":
    
    sample = input("Enter your comment: ")
    print(f"Result: {predcit(sample)}")
