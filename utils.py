import spacy

SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLES = ["parser", "ner"]


class Tokenizer:
    def __init__(self, minimum_len=2, maximum_len=15, lowercase=True,
                 output_lemma=True, use_stopwords=True, extra_stopwords=None,
                 alpha_only=True):

        if extra_stopwords is None:
            extra_stopwords = []
        self.nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLES, max_length=10 ** 7)

        for word in extra_stopwords:
            self.nlp.vocab[word].is_stop = True

        self.output_lemma = output_lemma
        self.lowercase = lowercase
        self.use_stopwords = use_stopwords
        self.minimum_len = minimum_len
        self.maximum_len = maximum_len
        self.alpha_only = alpha_only

    def _tokenize(self, doc):
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

    def _tokenize2(self, doc):
        tokenized = []
        for token in doc:
            if self.token_filter(token):
                tokenized.append(token.lemma_.lower())
        return tokenized

    def tokenize(self, text):
        return self._tokenize(self.nlp(text))

    def tokenize_pipe(self, texts):
        for doc, _ in self.nlp.pipe(texts, as_tuples=True, batch_size=1000, n_process=-1, disable=SPACY_DISABLES):
            yield self._tokenize2(doc), _

    def token_filter(self, token):
        return not ((self.alpha_only & ~token.is_alpha) |
                    (self.use_stopwords & token.is_stop) |
                    (self.minimum_len > len(token.lemma_) | len(token.lemma_) > self.maximum_len))
