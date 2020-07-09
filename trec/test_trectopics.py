from trec.trectopics import TrecTopics


def test_get_texts():
    assert False


def test_tokenize():
    assert False


def test_init():
    a = "tttt:  rr"
    print(a.replace("ee:", " "))


def test_get_title():
    pname = "f:/Runs/topics/"
    trectopics = TrecTopics(pname)
    trectopics.init()
    print(trectopics.get_title(51))
    print(trectopics.get_title(450))
    print(trectopics.get_narr(51))
    print(trectopics.get_narr(100))
    print(trectopics.get_narr(101))
    print(trectopics.get_narr(450))
    print(trectopics.get_narr(401))
    print(trectopics.get_title(251))
    print(trectopics.get_narr(251))
    print(trectopics.get_narr(296))
    print(trectopics.get_desc(296))


def test_get_desc():
    assert False


def test_get_narr():
    assert False


def test_total_topics():

    from gensim.models import Doc2Vec

    pname = "f:/Runs/topics/"
    trectopics = TrecTopics(pname)
    trectopics.init()

    model_fname = "F:/Models/Lemma_DM_2_doc2vec_trec_d1000_n5_w8_mc20_t12_e10_dm.model"

    d2v = Doc2Vec.load(model_fname)
    vocab_dict = d2v.wv.vocab

    trectopics.vectorize(vocab_dict=vocab_dict, include_title=True)
    print(trectopics.get_topic_vector(topic_no=51))


def test_utils():
    # lemmatized = "f:/Corpus/lemmatized_trec_all.dat"
    lemmatized = "f:/Corpus/new4.dat"
    from utils import read_content
    count = 0
    for text, meta in read_content(lemmatized):
        count += 1
        # if count is 10:
        #     break
        print(text, meta)


def test_verify():
    lemmatized = "f:/Corpus/lemmatized_trec_all.dat"
    from utils import verify_content
    duplicate, total_num = verify_content(lemmatized)
    print(total_num)
    print(duplicate)