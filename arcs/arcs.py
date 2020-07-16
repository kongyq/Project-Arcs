from math import ceil
from gensim.models.doc2vec import Doc2Vec
from trec.trecqrels import TrecQrels
from trec.trectopics import TrecTopics
from utils import sample_instance, sync_shuffle_lists

import numpy as np

DOC_RATIO = [0.7, 0.2, 0.1]
TOPIC_START = 51
TOPIC_END = 450


def get_hard_easy_neg_numbers(relevant_count):
    percentage = relevant_count / DOC_RATIO[0]
    return ceil(percentage * DOC_RATIO[1]), ceil(percentage * DOC_RATIO[2])


class Arcs:
    def __init__(self, model_file, qrels_path, topics_path):
        # assert sum(DOC_RATIO) is 1.0
        self.d2v = Doc2Vec.load(model_file)
        self.qrels = TrecQrels(qrels_path)
        self.qrels.init()

        self.topics = TrecTopics(topics_path)
        self.doc_list = self.d2v.docvecs.index2entity

        self.embedding_dim = self.d2v.vector_size
        self.vocab_size = len(self.d2v.wv.vocab)

        self.ts_doc_list = []
        self.ts_topic_list = []
        self.ts_label_list = []

    def create_training_set(self, shuffle=True):
        for topic_no in range(TOPIC_START, TOPIC_END + 1):
            relevant_docs = self.qrels.get_doc_list(topic_no, is_rel=True)
            irrelevant_docs = self.qrels.get_doc_list(topic_no, is_rel=False)

            # add all relevant docs into training set
            relevant_count = len(relevant_docs)
            self.ts_doc_list.extend(relevant_docs)
            self.ts_topic_list.extend([topic_no] * relevant_count)
            self.ts_label_list.extend([1] * relevant_count)

            # add some retrieved but irrelevant (hard neg) docs into training set
            hard_neg_count, easy_neg_count = get_hard_easy_neg_numbers(relevant_count)
            self.ts_doc_list.extend(self.qrels.get_random_docs(hard_neg_count, topic_no, is_rel=False))
            self.ts_topic_list.extend([topic_no] * hard_neg_count)
            self.ts_label_list.extend([0] * hard_neg_count)

            # add some totally irrelevant (easy neg) docs into training set
            exclusive_set = set(relevant_docs)
            exclusive_set.update(irrelevant_docs)
            self.ts_doc_list.extend(sample_instance(easy_neg_count, self.doc_list, exclusive_set=exclusive_set))
            self.ts_topic_list.extend([topic_no] * easy_neg_count)
            self.ts_label_list.extend([0] * easy_neg_count)

        if shuffle:
            self.ts_doc_list, self.ts_topic_list, self.ts_label_list = sync_shuffle_lists(self.ts_doc_list,
                                                                                          self.ts_topic_list,
                                                                                          self.ts_label_list)

    def get_training_set(self, use_topic_vector=False,
                         include_title=True, include_desc=False, include_narr=False, norm='l1'):
        if use_topic_vector:
            self.topics.init()
            self.topics.vectorize(vocab_dict=self.d2v.wv.vocab,
                                  include_title=include_title, include_desc=include_desc, include_narr=include_narr,
                                  norm=norm)
            return self.ts_doc_list, [self.topics.get_topic_vector(topic_no) for topic_no in self.ts_topic_list], self.ts_label_list
        else:
            return self.ts_doc_list, self.ts_topic_list, self.ts_label_list

    def get_docvecs(self):
        return [self.d2v.docvecs[doc] for doc in self.ts_doc_list]

    def get_topicvecs(self, use_topic_vector=False,
                      include_title=True, include_desc=False, include_narr=False, norm='l2'):
        if use_topic_vector:
            self.topics.init()
            self.topics.vectorize(vocab_dict=self.d2v.wv.vocab,
                                  include_title=include_title, include_desc=include_desc, include_narr=include_narr,
                                  norm=norm)
            return [self.topics.get_topic_vector(topic_no) for topic_no in self.ts_topic_list]
        else:
            return self.ts_topic_list

    def get_labels(self):
        return self.ts_label_list

    def init_topic_vecs(self, include_title=True, include_desc=False, include_narr=False, norm='l2'):
        self.topics.init()
        self.topics.vectorize(vocab_dict=self.d2v.wv.vocab,
                              include_title=include_title, include_desc=include_desc, include_narr=include_narr,
                              norm=norm)

    def training_set_iterator(self):
        for i in range(len(self.ts_doc_list)):
            yield ([self.d2v.docvecs[self.ts_doc_list[i]],
                   np.array(self.topics.get_topic_vector(self.ts_topic_list[i]), dtype=np.float32)], np.array(self.ts_label_list[i]))


    # def close(self):
    #     self.d2v=None
