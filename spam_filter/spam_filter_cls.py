from nltk.corpus import stopwords
import re
import pandas as pd

DEL_PUNCT = re.compile('[^\w\s]+')

try:
    stop_words = set(stopwords.words("english"))
except LookupError as e:
    print(e)
    print("Downloading packages...")
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

class SpamFilter:
    def __init__(self):
        self.cleaned_data = None

    def clean_data(self, data: str) -> list:
        data = data.lower() # make it lowercase
        data = re.sub(DEL_PUNCT, '', data) # remove punctuation
        data = [w for w in data.split() if w not in stop_words] # this meme was made by O(n^2) gang
        return data

    def fit(self, data: pd.DataFrame):
        """
        fit data
        :param X: first column is for phrases, second - for labels
        :return:
        """
        labels = data.iloc[:,1]
        phrases = data.iloc[:,0]

        self.cleaned_data = []

        for phrase in phrases:
            self.cleaned_data.append(
                self.clean_data(phrase)
            )


if __name__ == "__main__":
    print(SpamFilter().clean_data(input()))