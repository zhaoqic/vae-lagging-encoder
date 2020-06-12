
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ny': 7,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 64,
    'epochs': 100,
    'test_nepoch': 5,
    'train_num': 'datasets/g06n_data/g06n.num.train.npy',
    'val_num': 'datasets/g06n_data/g06n.num.valid.npy',
    'test_num': 'datasets/g06n_data/g06n.num.test.npy',
    'train_doc': 'datasets/g06n_data/g06n.doc.train.npy',
    'val_doc': 'datasets/g06n_data/g06n.doc.valid.npy',
    'test_doc': 'datasets/g06n_data/g06n.doc.test.npy',
    'train_label': 'datasets/g06n_data/g06n.label.train.npy',
    'val_label': 'datasets/g06n_data/g06n.label.valid.npy',
    'test_label': 'datasets/g06n_data/g06n.label.test.npy'
}
