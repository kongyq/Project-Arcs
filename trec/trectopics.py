from os import linesep

import logging

from gensim.corpora import TextDirectoryCorpus
from gensim.corpora.wikicorpus import tokenize

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

    def get_texts(self):

        inside_top = False
        inside_desc = False
        inside_narr = False
        topic_no = None
        title = None
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