from DataReader import DataReader
import nltk
'''
class Normalization(DataReader):
    def __init__(self, dataset):
        self.processed_dataset = dataset

    def tokenize(self, tokenizer=nltk.word_tokenize):
        for index, row in self.processed_dataset.iterrows():
            row["tweet"] = tokenizer(row["tweet"])
            row["tokenized_tweet"] = [] + row["tweet"]
            return row


    def stem(self, stemmer=nltk.PorterStemmer()):
        def stem_and_join(row):
            row["tweet"] = list(map(lambda str: stemmer.stem(str.lower()), row["tweet"]))
            return row

        self.processed_dataset = self.processed_dataset.apply(stem_and_join, axis=1)
'''


class Normalization(DataReader):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    def stem(self, stemmer=nltk.PorterStemmer()):
        def stem_and_join(row):
            row["text"] = list(map(lambda str: stemmer.stem(str.lower()), row["text"]))
            return row

        self.processed_data = self.processed_data.apply(stem_and_join, axis=1)

    def tokenize(self, tokenizer=nltk.word_tokenize):
        def tokenize_row(row):
            row["text"] = tokenizer(row["text"])
            row["tokenized_text"] = [] + row["text"]
            return row

        self.processed_data = self.processed_data.apply(tokenize_row, axis=1)