from DataReader import DataReader
from Normalization import Normalization
import re
import re as regex
import pandas as pd
import nltk
from collections import Counter


### Provides a pre-process for tweet messages.
### Replace emoticons, hash, mentions and urls for codes
### Correct long seguences of letters and punctuations
### Apply the Pattern part-of_speech tagger to the message
### Requires the Pattern library to work (http://www.clips.ua.ac.be/pages/pattern)


def remove_by_regex(tweets, regexp):
    tweets.loc[:, "tweet"].replace(regexp, "", inplace=True)
    return tweets


df_1 = DataReader('Data/tweeti-b.dist.tsv')
df1 = df_1.data_set

df_2 = DataReader('Data/downloaded.tsv')
df2 = df_2.data_set

df = pd.concat([df1, df2])
df = df.drop_duplicates(['tweet'])
df = remove_by_regex(df, regex.compile(r"http.?://[^\s]+[\s]?"))
df = remove_by_regex(df, regex.compile(r"@[^\s]+[\s]?"))
df = remove_by_regex(df, regex.compile(r"\s?[0-9]+\.?[0-9]*"))
for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     ".", "'", "--", "---", "#"]):
            df.loc[:, "tweet"].replace(remove, "", inplace=True)
tokenizer = nltk.word_tokenize
stemmer = nltk.PorterStemmer()

df["tokenized_tweet"] = "defaut value"

for index, row in df.iterrows():
    row["tokenized_tweet"] = tokenizer(row["tweet"])
    row["tweet"] = list(map(lambda str: stemmer.stem(str.lower()), tokenizer(row["tweet"])))

import os
if os.path.isfile("Data\wordlist.csv"):
    word_df = pd.read_csv("Data\wordlist.csv")
    word_df = word_df[word_df["occurrences"] > 3]
    wordlist = list(word_df.loc[:, "word"])

words = Counter()
for index, row in df.iterrows():
    words.update(row["tweet"])

stopwords=nltk.corpus.stopwords.words("english")
whitelist = ["n't", "not"]
for idx, stop_word in enumerate(stopwords):
    if stop_word not in whitelist:
        del words[stop_word]

word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common() if 3 < v < 500],
                            "occurrences": [v for k, v in words.most_common() if 3 < v < 500]},
                            columns=["word", "occurrences"])

word_df.to_csv("Data\wordlist.csv", index_label="idx")
wordlist = [k for k, v in words.most_common() if 3 < v < 500]


print(words.most_common(5))

#print(df)