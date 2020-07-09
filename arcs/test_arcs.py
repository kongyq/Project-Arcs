def test_create_training_set():
    assert False


def test_get_training_set():
    from arcs.arcs import Arcs
    from trec.trecqrels import TrecQrels
    model_file = "F:/Models/Lemma_DM_2_doc2vec_trec_d1000_n5_w8_mc20_t12_e10_dm.model"
    qrels_path = "f:/Runs/qrels/"
    topics_path = "f:/Runs/topics/"

    qrels = TrecQrels(qrels_path)
    qrels.init()

    arcs = Arcs(model_file=model_file, qrels_path=qrels_path, topics_path=topics_path)
    arcs.create_training_set(shuffle=True)
    ts_doc, ts_topic, ts_label = arcs.get_training_set()
    assert len(ts_doc) == len(ts_label)
    assert len(ts_doc) == len(ts_topic)

    # total_len = len(ts_doc)
    # rel_len = 0
    # for doc in ts_doc:
    #     if doc in qrels.get_doc_list(51):
    #         rel_len += 1
    #
    # label_len = 0
    # for label in ts_label:
    #     if label is 1:
    #         label_len += 1
    #
    # print(total_len, rel_len)
    # print(len(ts_label), label_len)
    # print(ts_doc)
    # print(ts_label)

    print(len(ts_doc))