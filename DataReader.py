import pandas as pd
from chart_studio.plotly import plotly
from plotly import graph_objs

class DataReader(object):

# Constructor
    def __init__(self,CSVdataSet):

        self.data_path = CSVdataSet
        self.data_set = list()

        self.reader()

    def reader(self):
        # Read dataSet
        df = pd.read_csv(self.data_path, sep='\t', header=0, names=["id", "usr_id", "sentiment", "tweet"])
        # delete not available tweets from dataSet
        df = df[df["tweet"] != 'Not Available']
        # substitute labels with objective and objective-OR-neutral for neutral
        df = df.replace('objective', 'neutral')
        df = df.replace('objective-OR-neutral', 'neutral')
        df = df.drop(["id", "usr_id"], axis=1)
        self.data_set = df


