from gensim.corpora import TextCorpus, TextDirectoryCorpus
from gensim.models.doc2vec import TaggedDocument

from trec.treccorpus import TrecCorpus

def test_get_texts():

    path = "F:/Corpus/trectest/"
    file = path + "fr881.dat"

    # with open(file, 'r') as fp:
    #     print(fp.read())

    trecc = TrecCorpus(path, dictionary={})
    for text, docno in trecc.get_texts():
        print(docno, text)
    # print(trecc.getstream())


def test_parse_file():

    def test():
        for i in range(0,10):
            yield i

    for i in test():
        print(i)
        break

    for i in test():
        print(i)
        break



def test_read_doc():
    a = "ddsad"
    b = [1,2,3,4,5]

class TaggedTrecDocument(object):
    def __init__(self, trec):
        self.trec = trec
        self.trec.metadata = True
    def __iter__(self):
        for content, (doc_id, title) in self.trec.get_texts():
            yield TaggedDocument(content, [doc_id])


def test_parse_text2222():
    # from trec.treccorpus import TrecCorpus
    pname = "f:/Corpus/trectest/"
    textt = TextDirectoryCorpus(pname, dictionary={}, metadata=True, lines_are_documents=True)

    documents = TaggedTrecDocument(textt)

    print(sum(1 for _ in documents))
    print(sum(1 for _ in documents))
    print(sum(1 for _ in documents))

def test_parse_text():
    # from trec.treccorpus import TrecCorpus
    pname = "f:/Corpus/trectest/"
    trecc = TrecCorpus(pname, dictionary={}, metadata=True)

    documents = TaggedTrecDocument(trecc)

    print(sum(1 for _ in documents))
    print(sum(1 for _ in documents))
    print(sum(1 for _ in documents))

    # total = 0
    # print()
    # for text, (docno, title) in trecc.get_texts():
    #     # print(docno)
    #     total += 1
    #     print(docno)
    #     # print(next(trecc.get_texts()))
    # print(total)

def test_traverse_all_docs():
    # pname = "f:/Corpus/TrecData/"
    pname = "f:/Corpus/trectest/"
    trecc = TrecCorpus(pname, dictionary={})
    count = 0
    for text, docno in trecc.get_texts():
        count += 1
        if count % 1000 == 0:
            print(docno, text)
            break

def test_save_to_file():
    pname = "f:/Corpus/trectest/"
    trecc = TrecCorpus(pname, dictionary={})
    sfile = "f:/Corpus/savetest.csv"
    trecc.save_to_file(sfile)