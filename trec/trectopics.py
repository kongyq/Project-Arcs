from gensim.corpora import TextDirectoryCorpus


class TrecTopics(TextDirectoryCorpus):
    def __init__(self, topics_path, min_depth=0, max_depth=None, metadata=True,
                 pattern=None, exclude_pattern=None, **kwargs):
        super(TrecTopics, self).__init__(topics_path, dictionary={}, metadata=metadata,
                                         min_depth=min_depth, max_depth=max_depth,
                                         pattern=pattern, exclude_pattern=exclude_pattern,
                                         lines_are_documents=True, **kwargs)

    def get_texts(self):

        for line in self.getstream():
            yield title, desc, narr

