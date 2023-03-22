# aggregate log event-level representation to seqence-level


evt_level_rep = np.load('./dataset/BGL/BGL_full.log_structured_0.8_rad_6.0.npz', allow_pickle=True)

x_train = evt_level_rep["x_train"][()]
y_train = evt_level_rep["y_train"]
x_test = evt_level_rep["x_test"][()]
y_test = evt_level_rep["y_test"]

# Aggregation
x_train_agg = []
for i in tqdm(range(x_train.shape[0])):
    fea = np.mean(x_train[i], axis = 0)
    x_train_agg.append(fea)
x_train_agg = np.array(x_train_agg)
x_train_agg.shape


x_test_agg = []
for i in tqdm(range(x_test.shape[0])):
    fea = np.mean(x_test[i], axis = 0)
    x_test_agg.append(fea)
x_test_agg = np.array(x_test_agg)
x_test_agg.shape

np.savez('./dataset/BGL/agg/BGL_word2vec_template_TFIDF_300d_agg.npz',x_train=x_train_agg, y_train = y_train, x_test = x_test_agg, y_test= y_test)