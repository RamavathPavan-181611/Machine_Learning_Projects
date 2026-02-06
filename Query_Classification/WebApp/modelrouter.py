import pickle


class GujaratiModel:
    def __init__(self):
        with open("models/Gujarati/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/Gujarati/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.model.predict(X)[0]


class HindiModel:
    def __init__(self):
        with open("models/Hindi/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/Hindi/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.model.predict(X)[0]


class EnglishModel:
    def __init__(self):
        with open("models/english/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/english/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.model.predict(X)[0]


class MultilingualModel:
    def __init__(self):
        with open("models/Multilingual/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/Multilingual/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.model.predict(X)[0]


class ModelHandler:
    def __init__(self, lang):
        if lang == "guj":
            self.model = GujaratiModel()
        elif lang == "hin":
            self.model = HindiModel()
        elif lang == "eng":
            self.model = EnglishModel()
        elif lang == "multi":
            self.model = MultilingualModel()
        else:
            raise ValueError("Unsupported language: choose 'guj', 'hin', 'eng', or 'multi'")

    def predict(self, query):
        return self.model.predict(query)
