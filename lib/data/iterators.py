
def get_infinity_iterator(train_data):
    train_iter = iter(train_data)
    while True:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data)
            continue
        yield batch
