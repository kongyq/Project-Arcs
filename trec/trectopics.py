from os import linesep

import logging

from gensim.corpora import TextDirectoryCorpus
from gensim.corpora.wikicorpus import tokenize

from scipy.sparse import csr_matrix
from scipy.special import softmax
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15


class TrecTopics(TextDirectoryCorpus):
    def __init__(self, topics_path, min_depth=0, max_depth=None, metadata=True,
                 pattern=None, exclude_pattern=None, **kwargs):
        super(TrecTopics, self).__init__(topics_path, dictionary={}, metadata=metadata,
                                         min_depth=min_depth, max_depth=max_depth,
                                         pattern=pattern, exclude_pattern=exclude_pattern,
                                         lines_are_documents=True, **kwargs)

        self.topics = {}
        self.topics_vecs = None
        self.topic_row_maps = {}
        self.oov = {}

    def get_texts(self):

        inside_top = False
        inside_desc = False
        inside_narr = False
        topic_no = None
        title = ""
        desc = ""
        narr = ""
        for line in self.getstream():
            if line.startswith("<top>"):
                inside_top = True
                continue

            if inside_desc:
                if line.startswith("<"):
                    inside_desc = False
                else:
                    desc += line + linesep
                    continue

            if inside_narr:
                if line.startswith("<"):
                    inside_narr = False
                else:
                    narr += line + linesep
                    continue

            if inside_top:
                if line.startswith("<num>"):
                    topic_no = line[line.find("Number:", len("<num>")) + len("Number:"):].strip()
                elif line.startswith("<title>"):
                    title = line[len("<title>"):].strip().replace("Topic:", "")
                elif line.startswith("<desc>"):
                    inside_desc = True
                elif line.startswith("<narr>"):
                    inside_narr = True
                elif line.startswith("</top>"):
                    inside_top = False
                    yield int(topic_no), self.tokenize(title), self.tokenize(desc), self.tokenize(narr)
                    title = ""
                    desc = ""
                    narr = ""

    def tokenize(self, text):
        return tokenize(text, TOKEN_MIN_LEN, TOKEN_MAX_LEN, lower=True)

    def init(self):
        for topic_no, title, desc, narr in self.get_texts():
            self.topics[topic_no] = {"title": title, "desc": desc, "narr": narr}

    def get_title(self, topic_no):
        return self.topics[topic_no]["title"]

    def get_desc(self, topic_no):
        return self.topics[topic_no]["desc"]

    def get_narr(self, topic_no):
        return self.topics[topic_no]["narr"]

    def total_topics(self):
        return len(self.topics)

    def indexedize(self, vocab_dict, topic_no, term_list):
        index_list = []
        for term in term_list:
            if term in vocab_dict:
                index_list.append(vocab_dict[term].index)
            else:
                self.oov.setdefault(topic_no, set()).add(term)
        return index_list

    def vectorize(self, vocab_dict, include_title=True, include_desc=False, include_narr=False, norm='l1'):
        assert include_title or include_desc or include_narr is True

        vector_length = len(vocab_dict)

        topic_row = [0]
        index_col = []
        freq_val = []

        for topic_no, title, desc, narr in self.get_texts():

            self.topic_row_maps.setdefault(topic_no, len(self.topic_row_maps))

            indexed_word_list = []
            if include_title:
                indexed_word_list.extend(self.indexedize(vocab_dict, topic_no, title))
            if include_desc:
                indexed_word_list.extend(self.indexedize(vocab_dict, topic_no, desc))
            if include_narr:
                indexed_word_list.extend(self.indexedize(vocab_dict, topic_no, narr))

            for index in indexed_word_list:
                index_col.append(index)
                freq_val.append(1)
            topic_row.append(len(index_col))

        self.topics_vecs = normalize(csr_matrix((freq_val, index_col, topic_row), dtype=int,
                                              shape=(len(topic_row)-1, vector_length)).toarray(),
                                     norm=norm, axis=1)

        # self.topics_vecs = csr_matrix((freq_val, index_col, topic_row), dtype=int,
        #                               shape=(len(topic_row)-1, vector_length)).toarray()

    def get_topic_vector(self, topic_no):
        assert 50 <= topic_no <= 450
        return self.topics_vecs[self.topic_row_maps[topic_no]]