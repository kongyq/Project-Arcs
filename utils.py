import spacy

SPACY_MODEL = "en_core_web_sm"


class Tokenizer:
    def __init__(self, minimum_len=2, maximum_len=15, lowercase=True,
                 output_lemma=True, use_stopwords=True, extra_stopwords=None,
                 alpha_only=True):

        if extra_stopwords is None:
            extra_stopwords = []
        self.nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"], max_length=10**7)

        # self.nlp.max_length = 10 ** 7
        for word in extra_stopwords:
            self.nlp.vocab[word].is_stop = True

        self.output_lemma = output_lemma
        self.lowercase = lowercase
        self.use_stopwords = use_stopwords
        self.minimum_len = minimum_len
        self.maximum_len = maximum_len
        self.alpha_only = alpha_only

    def tokenize(self, text):
        doc = self.nlp(text.strip())
        tokenized = []
        for token in doc:
            if self.alpha_only and not token.is_alpha:
                continue

            if self.output_lemma:
                word = token.lemma_
            else:
                word = token.text

            if self.lowercase:
                word = word.lower()

            if self.minimum_len <= len(word) <= self.maximum_len:
                if self.use_stopwords:
                    if not token.is_stop:
                        tokenized.append(word)
                else:
                    tokenized.append(word)
        return tokenized

    def tokenize_pipe(self, texts, as_tuples=True):
        tokenized = []
        for doc, content in self.nlp.pipe(texts, as_tuples=as_tuples, n_process=-1, disable=["parser", "ner"]):

            if self.output_lemma:
                yield [token.lemma_ for token in doc if self.token_filter(token)]
            else:
                yield [token.text for token in doc if self.token_filter(token)]

            for token in doc:
                if self.alpha_only and not token.is_alpha:
                    continue

                if self.use_stopwords and token.is_stop:
                    continue

                if self.output_lemma:
                    word = token.lemma_
                else:
                    word = token.text

                if self.lowercase:
                    word = word.lower()

                if self.minimum_len <= len(word) <= self.maximum_len:
                    if self.use_stopwords:
                        if not token.is_stop:
                            tokenized.append(word)
                    else:
                        tokenized.append(word)
            yield tokenized, content
            tokenized = []

    def token_filter(self, token):
        return not ((self.alpha_only & ~token.is_alpha) |
                    (self.use_stopwords & token.is_stop) |
                    (self.minimum_len > len(token.lemma_) | len(token.lemma_) > self.maximum_len))