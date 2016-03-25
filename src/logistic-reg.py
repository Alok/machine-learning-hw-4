
spam_data = sio.loadmat('../spam_dataset/spam_data.mat')
spam_train = spam_data['training_data']
spam_test = spam_data['test_data']

print("spam_train[1]: {}".format(len(spam_train[1])))

