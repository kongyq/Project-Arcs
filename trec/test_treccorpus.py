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



def test_parse_text():
    # from trec.treccorpus import TrecCorpus
    pname = "f:/Corpus/trectest/"
    trecc = TrecCorpus(pname, dictionary={}, metadata=True)
    total = 0
    for text, docno in trecc.get_texts():
        print(docno)
        total += 1
        # print(next(trecc.get_texts()))
    print(total)