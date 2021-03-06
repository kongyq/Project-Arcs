import spacy
import random
from gensim.models import Doc2Vec
from tensorflow.keras import models

SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLES = ["parser", "ner"]


class Tokenizer:
    """Customized Spacy tokenizer for tokenize and lemmatize TREC format documents.

    Argument
    --------
    * minimum_len: minimum word length will be kept
    * maximum_len: maximum word length will be kept
    * lowercase: If True, lower all words into lowercase
    * output_lemma: if True, lemmatize all tokens
    * use_stopwords: if True, remove all stopwords from documents
    * extra_stopwords: if a list, remove all terms in the list from documents
    * alpha_only: if True, only keep alphabetical tokens

    """

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
        return [token.lemma_.lower() for token in doc if self.token_filter(token)]

    def tokenize(self, text):
        """
        Tokenize text into a list of tokens, use single core, optimal for small documents or corpus.
        :param text: context string
        :return: a list of tokenized tokens
        """
        return self._tokenize(self.nlp(text))

    def tokenize_pipe(self, texts):
        """
        Return a generator that tokenize text into a list of tokens, use all cores,
        optimal for large corpus with long documents in it.
        :param texts: context string
        :return: a list of tokenized tokens
        """
        for doc, _ in self.nlp.pipe(texts, as_tuples=True, batch_size=128, n_process=-1, disable=SPACY_DISABLES):
            yield self._tokenize2(doc), _

    def token_filter(self, token):
        return not ((self.alpha_only & ~token.is_alpha) |
                    (self.use_stopwords & token.is_stop) |
                    (self.minimum_len > len(token.lemma_) | len(token.lemma_) > self.maximum_len))


def read_content(filename):
    """
    Read file that contains each document per row. use "|" as separator
    Document format as below:
        doc_no|title|text
    :param filename: filename of the file contains documents
    :return: a generator yield [tokens of text], (doc_no, title)
    """
    with open(filename, 'r') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            raw_data = line.split("|")
            yield raw_data[2].rstrip("\n").split(" "), (raw_data[0], raw_data[1])


def verify_content(filename):
    """
    Test purpose only!
    :param filename:
    :return:
    """
    docno_set = set()
    duplicate = []
    previous = ""
    for _, (docno, title) in read_content(filename):
        if docno is '':
            print(previous)
            break
        if docno in docno_set:
            duplicate.append(previous)
            print(previous)
            # break
        else:
            docno_set.add(docno)
        previous = docno
    return duplicate, len(docno_set)


def sample_instance(k, instance_list, exclusive_set=None):
    if exclusive_set is not None:
        container = list()
        for instance in instance_list:
            if instance not in exclusive_set:
                container.append(instance)
    else:
        container = instance_list
    assert len(container) >= 1
    if k > len(container):
        return random.choices(container, k)
    else:
        return random.sample(container, k)


def sync_shuffle_lists(*args):
    temp = list(zip(*args))
    random.shuffle(temp)
    return zip(*temp)


class ModelLoader:
    def __init__(self):
        self.d2v = None
        self.sbm = None

    def load_d2v(self, model_file):
        self.d2v = Doc2Vec.load(model_file)

    def load_sbm(self, model_file):
        self.sbm = models.load_model(model_file)

    def get_docs_list(self):
        assert self.d2v is not None
        return self.d2v.docvecs.index2entity

    def get_docvecs(self):
        assert self.d2v is not None
        return self.d2v.docvecs.vectors_docs

    def get_predict_vec_array(self):
        assert self.d2v is not None and self.sbm is not None
        return self.sbm.predict(self.get_docvecs())


# def save_predict_vecs(doc2vec_model, sbm_model, vecs_file):

def create_trec_runs(runs_file, rank_dict, topic_no, top=-1):
    count = 0
    sep = " "
    with open(runs_file, 'w') as fp:
        for key, value in rank_dict:
            if top is not -1 and count >= top:
                return
            fp.write(sep.join([str(topic_no), "Q0", key, str(count), str(value[0]),  "test", "\n"]))
            count += 1


def nonzero_entity_check(vector):
    zero_epsilon = 1e-10
    for entity in vector:
        return None


def docs_filter(unfiltered_doc_list, valid_prefix):
    filtered_doc_list = list()
    for doc in unfiltered_doc_list:
        for prefix in valid_prefix:
            if doc.startswith(prefix):
                filtered_doc_list.append(doc)
                break
    return filtered_doc_list
