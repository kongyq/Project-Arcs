from gensim.corpora import TextDirectoryCorpus
import random

class TrecQrels(TextDirectoryCorpus):
    """A class read and process TREC 1-8 Qrels.

    Notes
    -----
    TREC 1-8 Qrels can be founded at https://trec.nist.gov/data/reljudge_eng.html.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>> from trec.treccorpus import TrecCorpus
        >>>
        >>> path_to_qrels = datapath("/home/corpus/Runs/qrels/")
        >>>
        >>> qrels = TrecQrels(path_to_qrels)

    """
    def __init__(self, qrels_input, min_depth=0, max_depth=None,
                 pattern=None, exclude_pattern=None,
                 **kwargs):
        """

        Parameters
        ----------
        input : str
            Path to Qrels file/folder.
        min_depth : int, optional
            Minimum depth in directory tree at which to begin searching for files.
        max_depth : int, optional
            Max depth in directory tree at which files will no longer be considered.
            If None - not limited.
        pattern : str, optional
            Regex to use for file name inclusion, all those files *not* matching this pattern will be ignored.
        exclude_pattern : str, optional
            Regex to use for file name exclusion, all files matching this pattern will be ignored.
        lines_are_documents : bool, optional
            If True - each line is considered a document, otherwise - each file is one document.
        kwargs: keyword arguments passed through to the `TextCorpus` constructor.
            See :meth:`gemsim.corpora.textcorpus.TextCorpus.__init__` docstring for more details on these.

        """
        super(TrecQrels, self).__init__(qrels_input,
                                        dictionary={}, metadata=True,
                                        min_depth=min_depth, max_depth=max_depth,
                                        pattern=pattern, exclude_pattern=exclude_pattern,
                                        lines_are_documents=True, **kwargs)

        self.qrels = {}

        self.init()

    def get_texts(self):
        """Generate documents from corpus.

        Yields
        ------
        list of str
            Document as sequence of tokens (+ (doc_no, title) if self.metadata)

        """
        lines = self.getstream()
        for line in lines:
            seg = line.split()
            topic_no = seg[0]
            doc_no = seg[2]
            is_rel = seg[3]

            if self.metadata:
                yield doc_no, topic_no, is_rel
            else:
                yield doc_no

    def init(self):
        """
        Initialize the class by read all qrels files in the qrels_path

        :return: None
        """
        for doc_no, topic_no, is_rel in self.get_texts():
            self.qrels.setdefault(True if is_rel is "1" else False, {}).setdefault(topic_no, set()).add(doc_no)

    def get_doc_list(self, topic_no, is_rel=True):
        """
        Get relevant or irrelevant document list for specific topic number

        :param topic_no: topic number 51-450
        :param is_rel: If True : retrieve relevant document list.
                    If False: retrieve irrelevant document list.
        :return: A list of document numbers
        """
        return self.qrels[is_rel][topic_no]

    def contains(self, topic_no, doc_no, is_rel=True):
        """
        Check whether the document contained by the document list.

        :param topic_no: Topic number 51-450
        :param doc_no: Document number need to be checked
        :param is_rel: If True: Check relevant document list.
        If False: Check irrelevant document list.
        :return: True if contains.
        """
        return doc_no in self.qrels[is_rel][topic_no]

    def get_random_docs(self, k, topic_no, is_rel=False):
        """
        Return a k length random document numbers list by sampling the document list.

        :param k: length of returned list, if k is greater than total number of documents, choice documents
        with replacement. Else, choice documents without replacement.
        :param topic_no: Topic number 51-450
        :param is_rel: If True: sample relevant document list.
        If False: sample irrelevant document list.
        :return: A list of document numbers
        """
        total_docs = len(self.qrels[is_rel][topic_no])
        if k > total_docs:
            return random.choices(self.qrels[is_rel][topic_no], k)
        else:
            return random.sample(self.qrels[is_rel][topic_no], k)

    def get_doc(self, topic_no, is_rel=True):
        """
        Generator returns document number.

        :param topic_no: Topic number 51-450
        :param is_rel: If True: yield relevant document number.
        If False: yield irrelevant document number.
        :return:
        """
        for doc_no in self.qrels[is_rel][topic_no]:
            yield doc_no
