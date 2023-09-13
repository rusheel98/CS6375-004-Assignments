class Dataset:
  def __init__(self, dataset_name):
    self.dataset_name = dataset_name

  def load_data(self):
    self.data = pd.read_csv(self.dataset_name)
    return self.data

  def showData(self):
    print(self.data.head(4))

  def train_test_split(self, dataframe, split=0.8):
    train = dataframe.sample(frac=split,random_state=200)
    test = dataframe.drop(train.index)
    return train, test