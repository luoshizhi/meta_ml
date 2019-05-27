import warnings
from sklearn import metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.utils import column_or_1d
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import sys
warnings.filterwarnings("ignore")


def evalu(y_test, y_pred, lable_type="regress"):
    if lable_type == "regress":
        print("explained_variance_score:",
              explained_variance_score(y_test, y_pred))
        print("mean_absolute_error:", mean_absolute_error(y_test, y_pred))
        print("mean_squared_error:", mean_squared_error(y_test, y_pred))
        print("median_absolute_error:", median_absolute_error(y_test, y_pred))
        print("r2_score:", r2_score(y_test, y_pred))
    else:
        print(metrics.classification_report(y_test, y_pred))
        if lable_type == "binary":
            print("roc_auc_score:", metrics.roc_auc_score(y_test, y_pred))


def filter_x_y(x_df, y_df):

    y_data = y_df[y_df.isnull().values == False]
    x_tmp = x_df.loc[y_df.isnull().values == False, :]
    x_tmp = pd.DataFrame(x_tmp, dtype=np.float)
    x_data = x_tmp.loc[:, (x_tmp == 0).sum(axis=0)/x_tmp.shape[0] < 0.5]
    if len(y_data.unique()) * 20 > x_data.shape[0]:
        lable_type = "regress"
    elif len(y_data.unique()) == 2:
        lable_type = "binary"
    else:
        lable_type = "multi"
    if lable_type == "regress":
        try:
            y_data = pd.DataFrame(y_data, dtype=np.float)
        except ValueError:
            return None, None, None
    return x_data, y_data, lable_type


class MlModel():
    def __init__(self, lable_type="multi"):
        self.lable_type = lable_type

    def _curFname(self):
        print('[{0}]'.format(sys._getframe().f_back.f_code.co_name))

    def runLogistic(self, X_train, y_train, traincv=False):
        self._curFname()
        model = LogisticRegression()
        if traincv:
            tuned_parameters = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    'penalty': ['l1', 'l2']
                                }
            model = GridSearchCV(model, tuned_parameters, cv=10)
        model.fit(X_train, y_train)
        return model

    def runDecisionTree(self, X_train, y_train, traincv=False):
        self._curFname()
        model = DecisionTreeRegressor(
        ) if self.lable_type == "regress" else DecisionTreeClassifier()
        if traincv:
            tuned_parameters = {
                'max_features': ["auto", "sqrt", "log2"],
                'min_samples_leaf': range(1, 100, 5),
                'max_depth': range(1, 50, 5)
                                }
            model = GridSearchCV(model, tuned_parameters, cv=10)
        model.fit(X_train, y_train)
        return model

    def runRandomForest(self, X_train, y_train, traincv=False):
        self._curFname()
        model = RandomForestRegressor(
        ) if self.lable_type == "regress" else RandomForestClassifier()
        if traincv:
            tuned_parameters = {
                    'min_samples_leaf': range(10, 100, 10),
                    'n_estimators': range(10, 100, 10),
                    'max_features': ['auto', 'sqrt', 'log2']
                                }
            model = GridSearchCV(model, tuned_parameters, cv=10)
        model.fit(X_train, y_train)
        return model

    def runXgboost(self, X_train, y_train, traincv=False):
        self._curFname()
        if self.lable_type == "regress":
            model = XGBRegressor(max_depth=5, objective='reg:gamma')
        elif self.lable_type == "binary":
            model = XGBClassifier(objective='binary:logistic')
        else:
            model = XGBClassifier(objective='multi:softprob')
        model.fit(X_train, y_train)
        return model

    def evaluation(self, X_test, y_test, model):
        if self.lable_type == "regress":
            y_pred = model.predict(X_test)
            # model.score(X_test, y_pred)
        else:
            y_prob = model.predict_proba(X_test)
            y_pred = y_prob.argmax(axis=1)
            if self.lable_type == "multi":
                print("log_loss:", metrics.log_loss(y_test, y_prob))
        evalu(y_test, y_pred, self.lable_type)

    def select_features(self, X_train, y_train, X_test):
        print("select_features...")
        if self.lable_type == "regress":
            model = GradientBoostingRegressor()
        else:
            model = GradientBoostingClassifier()
        gbdt_RFE = RFE(model, int(0.8*X_train.shape[1]))
        gbdt_RFE.fit(X_train, y_train)
        return (X_train.loc[:, gbdt_RFE.support_],
                X_test.loc[:, gbdt_RFE.support_])


def process_data(data, index=8):
    labelencoder = LabelEncoder()
    y_title_index = index
    y_title = data.loc[0, y_title_index]
    print("Training model for {}".format(y_title))
    x_df = data.loc[1:, 31:]
    y_df = data.loc[1:, y_title_index]
    x_data, y_data, lable_type = filter_x_y(x_df, y_df)
    if x_data is None:
        sys.stderr.write("skip {} {}\n".format(index, y_title))
        return None
    plot_sample_dis(y_data, y_title, lable_type)
    if lable_type == "multi" or lable_type == "binary":
        y_data = labelencoder.fit_transform(y_data)
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=4)
    y_train = column_or_1d(y_train)
    return X_train, X_test, y_train, y_test, lable_type


def run_model_pipeline(X_train, X_test, y_train, y_test, lable_type):
    model = MlModel(lable_type)
    if lable_type != "regress":
        DT = model.runLogistic(X_train, y_train, traincv=True)
        model.evaluation(X_test, y_test, DT)
    DT = model.runDecisionTree(X_train, y_train, traincv=True)
    model.evaluation(X_test, y_test, DT)
    DT = model.runRandomForest(X_train, y_train, traincv=True)
    model.evaluation(X_test, y_test, DT)
    DT = model.runXgboost(X_train, y_train, traincv=True)
    model.evaluation(X_test, y_test, DT)


def plot_sample_dis(y_data, y_title, lable_type):
    pyplot.figure(figsize=(15.0, 10.0))
    if lable_type == "multi" or lable_type == "binary":
        sns.countplot(y_data)
    else:
        sns.distplot(y_data, hist=True, kde=True)
    pyplot.xlabel(y_title+" sample")
    pyplot.ylabel("value")
    pyplot.savefig(y_title+"_sample.png")


def main():
    data = pd.read_table(
            "./merge.metaphlan_tables.tree.merge.metadata.new.noLD16_2",
            header=None)
    for i in range(1, 30):
        data_sets = process_data(data, index=i)
        if data_sets is None:
            continue
        X_train, X_test, y_train, y_test, lable_type = data_sets
        print(i, lable_type)
        print("using {} samples; {} features to training".format(
            X_train.shape[0], X_train.shape[1]))
        print("using {} samples; {} features to testing".format(
            X_test.shape[0], X_train.shape[1]))
        model = MlModel(lable_type)
        run_model_pipeline(X_train, X_test, y_train, y_test, lable_type)
        X_train, X_test = model.select_features(X_train, y_train, X_test)
        print("using {} samples; {} features to training".format(
            X_train.shape[0], X_train.shape[1]))
        print("using {} samples; {} features to testing".format(
            X_test.shape[0], X_train.shape[1]),)
        run_model_pipeline(X_train, X_test, y_train, y_test, lable_type)
        print("===Done!")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
