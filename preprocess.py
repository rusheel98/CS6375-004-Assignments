class Preprocess:
  def __init__(self, dataframe):
    self.data = dataframe

  def removeNulls(self):
    self.data.dropna(inplace=True)
    # print(self.data.shape)
    self.data.head(3)
    return self.data

  def removeDuplicates(self):
    self.data.drop_duplicates(inplace=True)
    # print(self.data.shape)
    return self.data

  def categoricalToNumerical(self, column_name):
    # print("before",self.data.shape)
    dummies = pd.get_dummies(self.data[column_name])
    self.data = pd.concat([self.data,dummies], axis='columns')
    # print("after",self.data.shape)
    return self.data

  def normalization(self):
    self.data = self.data.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(4))
    # print(self.data.shape)
    return self.data

  def dropColumns(self, column_names):
    # print("before",self.data.shape)
    self.data = self.data.drop(column_names, axis=1)
    # print("after",self.data.shape)
    return self.data

  def reorderColumns(self,column_name):
    self.data = self.data[[col for col in self.data if col not in column_name] + column_name]
    # self.data.head(3)
    return self.data