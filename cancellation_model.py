import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle

features = ['day_distance', 'length_of_stay',
            'commission', 'country_code', 'provider_code_encoded']
target = 'cancelled'


class CancellationModel:
    def __init__(self, _model_settings={}):
        self.model_settings = _model_settings

    def preprocess(self, _preprocess_data_df, _inplace=True):
        if _inplace:
            df = _preprocess_data_df
        else:
            df = _preprocess_data_df.copy()

        for c in df[features].columns:
            col_type = df[c].dtype
            if col_type == 'object' or col_type.name == 'category':
                print(c)
                df[c] = df[c].astype(
                    'category')
        return df

    def train(self, _train_data, _train_params={}):

        self.preprocess(_train_data, _inplace=True)
        train_data, validation_data = train_test_split(
            _train_data, test_size=0.3, random_state=1)
        train_data.reset_index(drop=True, inplace=True)
        validation_data.reset_index(drop=True, inplace=True)

        d_train = lgb.Dataset(train_data[features], label=train_data[target])
        if _train_params:
            train_params = _train_params
        elif self.model_settings:
            train_params = self.model_settings["train_params"]
        else:
            print("Cannot train model: no parameters specified")
            return None

        self.model = lgb.train(train_params,
                               d_train, self.model_settings["epocs"])

        train_y_pred, train_y_true = self.predict(
            train_data, _preprocess=False)
        val_y_pred, val_y_true = self.predict(
            validation_data, _preprocess=False)
        self.train_score = self.score(train_y_true, train_y_pred)
        self.val_score = self.score(val_y_true, val_y_pred)
        print("trained scores: ", self.train_score, self.val_score)
        return self.model

    def dump(self, _fileName):
        print("dumping")
        self.model.save_model(_fileName)
        file_to_save = open(_fileName, 'wb')
        file_to_save.write(
            bytearray(str(self.train_score)[:10].encode("ascii")))
        file_to_save.write(bytearray(str(self.val_score)[:10].encode("ascii")))
        pickle.dump(self.model, file_to_save)
        file_to_save.close()

    def load(_fileName):
        print("loading")
        current_model = CancellationModel()
        # current_model.model = lgb.Booster(model_file=_fileName)
        file_to_save = open(_fileName, 'rb')
        current_model.train_score = float(file_to_save.read(10))
        current_model.val_score = float(file_to_save.read(10))
        print("loaded scores = ", current_model.train_score,
              current_model.val_score)
        current_model.model = pickle.load(file_to_save)
        file_to_save.close()
        return current_model

    def predict(self, _predict_data, _preprocess=True):
        if _preprocess:
            self.preprocess(_predict_data, _inplace=True)
        y_pred = self.model.predict(_predict_data[features])
        return y_pred, _predict_data[target]

    def score(self, _y_true, _y_pred):
        # if _y_true and _y_pred:
        y_pred = _y_pred
        y_true = _y_true
        return roc_auc_score(y_true, y_pred)


def load_training_data(_fileName=""):
    if not _fileName:
        # df = pd.read_csv('https://fh-public.s3.eu-west-1.amazonaws.com/ml-engineer/cancellation_dataset.csv')
        df = pd.read_csv('cancellation_dataset.csv')
        train, _ = train_test_split(df, test_size=0.3, random_state=0)

    return train.reset_index(drop=True)


def load_prediction_data(_fileName=""):
    if not _fileName:
        # df = pd.read_csv('https://fh-public.s3.eu-west-1.amazonaws.com/ml-engineer/cancellation_dataset.csv')
        df = pd.read_csv('cancellation_dataset.csv')
        _, test = train_test_split(df, test_size=0.3, random_state=0)
    return test.reset_index(drop=True)
