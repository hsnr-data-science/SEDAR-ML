import json
from enum import Enum
from typing import Union

import pandas as pd
import math
from scipy import stats
import numpy as np
import datetime
import sys
import base64
import io
import time
import nltk
from nltk.corpus import stopwords

from common.data.dataset import TargetFeatureType

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import collections
import scipy
from sklearn.feature_selection import f_classif, mutual_info_classif
from functools import reduce


class ReadMode(Enum):
    READ_CSV_FROM_FILE = 1
    READ_CSV_FROM_BASE64 = 2
    READ_FROM_DATAFRAME = 3


class DataProfiler():

    ##########################################################################################################
    ########################################## Function Definition ###########################################
    ##########################################################################################################

    def __init__(self, dataset_name, target_label, target_feature_type: Union[str, TargetFeatureType]):
        self.dataset_name = dataset_name
        self.class_label = target_label
        self.target_feature_type = TargetFeatureType[target_feature_type] if isinstance(target_feature_type, str) else target_feature_type
        self.nr_total_features = 0
        self.nr_analyzed_features = 0
        self.df = ''
        self.df_complete = ''
        self.column_names_list = ''
        self.miss_value = ''
        self.drop_cols = ''
        self.json_data = {}
        self.numerical_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.unstructured_features = []
        self.collection_datasets = ''
        self.json_data["info"] = {}
        self.json_data["features"] = {}
        self.json_data["info"]["datasetName"] = self.dataset_name
        self.json_data["info"]["targetLabel"] = self.class_label
        self.json_data["info"]["targetFeatureType"] = target_feature_type if isinstance(target_feature_type, str) else target_feature_type.value

    # Return a new dataset after dropping missing values. And return Number of missing values in each column.
    def handle_missing_values(self, df: pd.DataFrame):
        print('size of df before dropping missing values: ' + str(len(df)))
        print("Number of missing values in each column:")
        miss_value = (df.isnull().sum())
        print(miss_value)
        null_stats = miss_value.sum()
        print("Number of missings data in the whole dataset: " + str(null_stats))

        # drop the column if this col has missing values more than half of the whole dataset
        drop_cols = list()
        num_rows = len(df)
        for index, val in enumerate(miss_value):
            if val > num_rows / 4:
                drop_cols.append(df.columns[index])
        print("Dropped columns: " + str(drop_cols))
        df = df.drop(drop_cols, axis=1)

        # Drop the rows even with single NaN or single missing values.
        df = df.dropna()
        print('size of df after dropping missing values: ' + str(len(df)))
        return (df, miss_value, drop_cols)

        # Calculate Interquartile range and Quartiles

    def iqr_cal(self, df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q2 = df_in[col_name].quantile(0.5)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        q0 = q1 - (1.5 * iqr)
        q4 = q3 + (1.5 * iqr)
        return q0, q1, q2, q3, q4, iqr

    # Detect the outliers in a column using interquartile range
    def detect_outlier(self, df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        # print("fence_low: "+str(fence_low)+", fence_high: "+str(fence_high))
        df_out = df_in[col_name].loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)]
        return df_out.to_numpy()

    # Return correlation between second argument(target col) and current column
    def corr_cal(self, df, col_name):
        return (df[col_name].corr(df[self.class_label]))

    # Return minimum order in current column
    def min_orderm_cal(self, df, col_name):
        if abs(df[col_name].min()) == 0:
            return -math.inf
        else:
            return (round(math.log10(abs(df[col_name].min()))))

    # Return maximum order in current column
    def max_orderm_cal(self, df, col_name):
        if abs(df[col_name].max()) == 0:
            return -math.inf
        else:
            return (round(math.log10(abs(df[col_name].max()))))

    # Return number of categories/levels in a column for categorical feature
    def freq_counts(self, df, col_name):
        res = df[col_name].value_counts()
        val_list = res.tolist()
        index_list = res.index.tolist()
        return index_list, val_list, len(index_list)

    # Retun how big is the imbalance of feature (ratio between most popular and least popular)
    def imbalance_test(self, df, col_name):
        res = df[col_name].value_counts().tolist()
        if min(res) == 0:
            return math.inf
        return max(res) / min(res)

    # Chi-square Test of Independence using scipy.stats.chi2_contingency
    # The H0 (Null Hypothesis): There is no relationship between variable one and variable two.
    # Null hypothesis is rejected when the p-value is less than 0.05
    def chisq_correlated_cal(self, df, col_name):
        crosstab = pd.crosstab(df[self.class_label], df[col_name])
        res = stats.chi2_contingency(crosstab)
        p = res[1]
        # return p-value, We can reject the null hypothesis as the p-value is less than 0.05
        if p < 0.05:
            ifCorr = 'True'
        else:
            ifCorr = 'False'
        return (p, ifCorr)

    def text_statistics(self, df, col_name):
        feature = df[col_name]
        min_vocab = 1000
        max_vocab = 0
        vocab_size = 0
        print("total_text")
        new_words = []
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        #print(stopwords.words())
        #print(type(stopwords.words()))
        #exit()

        for i, txt in enumerate(feature):
            text_tokens = word_tokenize(txt)
            vocab_size_doc = len(text_tokens)
            vocab_size += vocab_size_doc
            if vocab_size_doc > max_vocab:
                max_vocab = vocab_size_doc
            if vocab_size_doc < min_vocab:
                min_vocab = vocab_size_doc
            new_words += tokenizer.tokenize(txt)
            #new_words = list(set(new_words))


        print("tokens_without_sw")
        print(len(new_words))
        tokens_without_sw = list(reduce(lambda x,y : filter(lambda z: z!=y,x) ,stopwords.words('english'),new_words))
        # now = datetime.datetime.now()
        # for i, word in enumerate(new_words):
        #     if word not in stopwords.words():
        #         tokens_without_sw.append(word)
        #     if i/len(new_words) >= 0.1:
        #         then = datetime.datetime.now()
        #         total = then-now
        #         print(i)
        #         print('estimated remaining time: ', total, ' seconds')
        #tokens_without_sw = [word for word in new_words if not word in stopwords.words()]

        print("relative vocabulary")
        # relative vocabulary
        nm = len(tokens_without_sw)
        relative_vocab = vocab_size / nm

        print("vocabulary concentration")
        # vocabulary concentration
        elements_count = collections.Counter(tokens_without_sw)
        vocab_freq_lst = sorted(tokens_without_sw, key=lambda x: -elements_count[x])
        vocab_freq_lst = pd.unique(vocab_freq_lst).tolist()
        top_10 = []
        for i in range(10):
            top_10.append(vocab_freq_lst[i])
        n_top = 0
        for i, x in enumerate(top_10):
            n_top += tokens_without_sw.count(x)
        vocab_concentration = n_top / len(tokens_without_sw)

        print("entropy")
        # entropy
        data = pd.Series(tokens_without_sw)
        counts = data.value_counts()
        entropy = scipy.stats.entropy(counts)

        return vocab_size, relative_vocab, vocab_concentration, entropy, min_vocab, max_vocab

    # claculate the monotonous filtering for the numerical features
    def monotonous_filtering_numerical(self, df, col_name):
        feature = df[col_name]
        mean = feature.mean(axis=0)
        std = feature.std(axis=0)
        fence_1 = mean - std
        fence_2 = mean + std
        total_number_of_values = len(feature)
        number_of_features_inside_fences = 0
        for idx, i in enumerate(feature):
            if fence_1 <= i <= fence_2:
                number_of_features_inside_fences += 1
        percentage_of_monotonic_values = number_of_features_inside_fences / total_number_of_values
        return percentage_of_monotonic_values

    # claculate the monotonous filtering for the categorical features
    def monotonous_filtering_categorical(self, df, col_name):
        feature = df[col_name]
        levels = []
        frequency = []
        for idx, i in enumerate(feature):
            if i not in levels:
                levels.append(i)
                frequency.append(1)
            else:
                frequency[levels.index(i)] += 1
        num_highest_levels = math.ceil(0.1 * len(frequency))
        highest_levels = []
        freq_level = zip(frequency, levels)
        freq_level_sorted = sorted(freq_level, reverse=True)
        for i in range(num_highest_levels):
            highest_levels.append(freq_level_sorted[i][1])
        total_number_of_values = len(feature)
        number_of_values_in_highest_levels = 0
        for idx, i in enumerate(feature):
            if i in highest_levels:
                number_of_values_in_highest_levels += 1
        percentage_of_monotonic_values = number_of_values_in_highest_levels / total_number_of_values
        return percentage_of_monotonic_values

    # Shapiro-Wilk test for normality.
    # H0 (Null Hypothesis): Normal distributed.
    # p value less than 0.05 means that null hypothesis is rejected.
    def shapiro_test_normality(self, df, col_name):
        shapiro_test = stats.shapiro(df[col_name])
        if shapiro_test[1] < 0.05:
            return False
        else:
            return True

    # Kolmogorov Smirnov test for Exponential distribution.
    # H0 (Null Hypothesis): Exponentially distributed.
    # p value less than 0.05 means that null hypothesis is rejected.
    def ks_test_exponential(self, df, col_name):
        ks_test = stats.kstest(df[col_name], 'expon')
        if ks_test.pvalue < 0.05:
            return False
        else:
            return True

    # Converts numpy datatypes into python default datatypes
    @staticmethod
    def datatype_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()


    # Read pandas dataframe and handle missing values
    def process_pandas_df(self, mode: ReadMode, dataset_path=None, dataset_string=None, dataset_df=None):
        missing_values = ["n/a", "na", "--", "NA", "?"," ?", "", " ", "NAN", "NaN"]
        if mode == ReadMode.READ_CSV_FROM_FILE:
            self.df = pd.read_csv(dataset_path + "/" + self.dataset_name, sep=",", na_values=missing_values)
        elif mode == ReadMode.READ_CSV_FROM_BASE64:
            decoded = base64.b64decode(dataset_string)
            self.df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep=",", na_values=missing_values)
        elif mode == ReadMode.READ_FROM_DATAFRAME:
            if dataset_df is None:
                return "processing failed"
            self.df = dataset_df
        self.column_names_list = list(self.df.columns)
        print(self.df.dtypes)

        self.df_complete = pd.DataFrame()
        self.df_complete = pd.concat([self.df_complete, self.df], axis=1)

        self.nr_total_features = self.df_complete.shape[1]
        self.json_data["info"]["nrTotalFeatures"] = self.nr_total_features - 1  # Do not count class label

        # Check if the target label provided by the user exists
        if not self.class_label in self.column_names_list:
            return "processing failed"
        # handle_missing_values
        self.json_data["info"]["observations"] = self.df.shape[0]
        (self.df, self.miss_value, self.drop_cols) = self.handle_missing_values(self.df)
        self.json_data["info"]["analyzedObservations"] = self.df.shape[0]
        self.nr_analyzed_features = self.df.shape[1] - 1  # Do not count class label
        self.json_data["info"]["nrAnalyzedFeatures"] = self.nr_analyzed_features
        return ("processing success")

        # Identify indices of numerical, categorical features
    def process_feature_annotation_list(self, feature_annotations: str):
        print("DEBUG: feature_annotations =", feature_annotations)
        print("DEBUG: class_label =", self.class_label)
        # Splitte Annotationen robust in Paare
        import re
        matches = re.findall(r'([^,]+),([A-Z])', feature_annotations)
        feature_names = [name.strip() for name, typ in matches]
        feature_types = [typ.strip() for name, typ in matches]
        if 'T' not in feature_types:
            return "processing failed: no feature annotated with class label 'T' in annotation list"
        # Setze die Feature-Indices wie bisher
        for i, typ in enumerate(feature_types):
            if typ == 'N':
                self.numerical_features.append(i)
            elif typ == 'C':
                self.categorical_features.append(i)
            elif typ == 'D':
                self.datetime_features.append(i)
            elif typ == 'U':
                self.unstructured_features.append(i)
            elif typ == 'T':
                print("Identified Class Label in annotation list")
            else:
                print("Please recheck the feature annotation list")
                column_list = self.column_names_list
                for column in self.drop_cols:
                    column_list.remove(column)
                return column_list[i]
        return "parsing successfully completed"

    # Calculate ratios
    def calculate_ratios(self):
        # Calculate ratios
        nr_numeric_features = len(self.numerical_features)
        nr_categorical_features = len(self.categorical_features)
        nr_datetime_features = len(self.datetime_features)
        nr_unstructured_features = len(self.unstructured_features)
        self.json_data["info"]["numericalRatio"] = float("{:.2f}".format(nr_numeric_features / self.nr_total_features))
        self.json_data["info"]["categoricalRatio"] = float(
            "{:.2f}".format(nr_categorical_features / self.nr_total_features))
        self.json_data["info"]["datetimeRatio"] = float("{:.2f}".format(nr_datetime_features / self.nr_total_features))
        self.json_data["info"]["unstructuredRatio"] = float("{:.2f}".format(nr_unstructured_features / self.nr_total_features))

    # Calculate parameters for numerical features and add it to json
    def analyse_numerical_features(self):
        print("Analysing numerical features")
        # Calculate parameters for numerical features and add it to json
        self.json_data["features"]["numericalFeatures"] = {}
        self.json_data["info"]["analyzedFeatures"] = []
        self.json_data["info"]["discardedFeatures"] = []
        feature = ""
        numericalFeatures = pd.DataFrame()
        for i in range(len(self.numerical_features)):
            if self.df_complete.columns[self.numerical_features[i]] in self.df.columns:
                numericalFeatures = pd.concat([numericalFeatures, self.df[self.df_complete.columns[self.numerical_features[i]]]],
                                          axis=1)
        if not (len(numericalFeatures) == 0):
            anova_f1 = f_classif(numericalFeatures, self.df[self.class_label])[0]
            anova_pvalue = f_classif(numericalFeatures, self.df[self.class_label])[1]
            if self.target_feature_type in [TargetFeatureType.BINARY, TargetFeatureType.CATEGORICAL]:
                y = self.df[self.class_label]
                mi = mutual_info_classif(numericalFeatures, self.df[self.class_label], random_state=42)
            else:
                # For regression problems, we can't calculate mutual information
                mi = None
        counter = 0
        try:
            for column_nr in self.numerical_features:
                feature = self.column_names_list[column_nr]
                if (feature not in self.drop_cols):
                    self.json_data["info"]["analyzedFeatures"].append(feature)
                    self.json_data["features"]["numericalFeatures"][feature] = {}
                    # Implement the monotonous filtering
                    self.json_data["features"]["numericalFeatures"][feature]['monotonousFiltering'] = self.monotonous_filtering_numerical(self.df, feature)
                    # Assign the f1 value from the anova:
                    self.json_data["features"]["numericalFeatures"][feature]['anovaF1'] = anova_f1[counter]
                    # Assign the p value from the anova:
                    self.json_data["features"]["numericalFeatures"][feature]['anovaPvalue'] = anova_pvalue[counter]
                    # Assign the mutual information for the feature
                    if mi is not None:
                        self.json_data["features"]["numericalFeatures"][feature]['mutualInfo'] = mi[counter]
                    counter = counter + 1
                    # Calculate missing values
                    self.json_data["features"]["numericalFeatures"][feature]['missingValues'] = self.miss_value[
                        feature]
                    # Calculate min and max values
                    self.json_data["features"]["numericalFeatures"][feature]['minValue'] = self.df[feature].min()
                    self.json_data["features"]["numericalFeatures"][feature]['maxValue'] = self.df[feature].max()
                    # Calculate min order and max order
                    self.json_data["features"]["numericalFeatures"][feature]['minOrderm'] = self.min_orderm_cal(
                        self.df, feature)
                    self.json_data["features"]["numericalFeatures"][feature]['maxOrderm'] = self.max_orderm_cal(
                        self.df, feature)
                    # Calculate the correlation between selected feature and target feature.
                    #if 'numeric' in self.target_feature_type or 'numeric' in self.target_feature_type:
                    #    self.json_data["features"]["numericalFeatures"][feature]['correlation'] = self.corr_cal(self.csv_data, feature)
                    #elif 'categoric' in self.target_feature_type or 'categoric' in self.target_feature_type or 'binary' in self.target_feature_type or 'binary' in self.target_feature_type:
                    #    (pval, chisq_correlated) = self.chisq_correlated_cal(self.csv_data, feature)
                    #    self.json_data["features"]["numericalFeatures"][feature]['correlation'] = {}
                    #    self.json_data["features"]["numericalFeatures"][feature]['correlation']['chisqCorrelated'] = chisq_correlated
                    #    self.json_data["features"]["numericalFeatures"][feature]['correlation']['pVal'] = pval
                    # Calculate IQR and Quartiles
                    q0, q1, q2, q3, q4, iqr = self.iqr_cal(self.df, feature)
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles'] = {}
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles']['q0'] = q0
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles']['q1'] = q1
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles']['q2'] = q2
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles']['q3'] = q3
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles']['q4'] = q4
                    self.json_data["features"]["numericalFeatures"][feature]['quartiles']['iqr'] = iqr
                    # Calculate outlier info
                    outlier_list = self.detect_outlier(self.df, feature)
                    self.json_data["features"]["numericalFeatures"][feature]['outliers'] = {}
                    self.json_data["features"]["numericalFeatures"][feature]['outliers']['number'] = len(outlier_list)
                    self.json_data["features"]["numericalFeatures"][feature]['outliers'][
                        'actualValues'] = outlier_list
                    # Distribution Check
                    self.json_data["features"]["numericalFeatures"][feature]['distribution'] = {}
                    normal_distrn_bool = self.shapiro_test_normality(self.df, feature)
                    self.json_data["features"]["numericalFeatures"][feature]['distribution']['normal'] = normal_distrn_bool
                    self.json_data["features"]["numericalFeatures"][feature]['distribution']['exponential'] = self.ks_test_exponential(self.df, feature)
                    if normal_distrn_bool:
                        self.json_data["features"]["numericalFeatures"][feature]['distribution']['skewness'] = stats.skew(self.df[feature])
                else:
                    self.json_data["info"]["discardedFeatures"].append(feature)
                    print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")
            return ("analysis successfully completed")
        except TypeError:
            print("Numeric Feature Analysis Terminated")
            print("Please recheck feature type of feature: " + feature)
            return (feature)

    # Calculate parameters for categorical features and add it to json
    def analyse_categorical_features(self):
        print("Analysing categorical features")
        # Calculate parameters for categorical features and add it to json
        self.json_data["features"]["categoricalFeatures"] = {}
        categorical_features = pd.DataFrame()
        for i in range(len(self.categorical_features)):
            if self.df_complete.columns[self.categorical_features[i]] in self.df.columns:
                self.df[self.df_complete.columns[self.categorical_features[i]]]= self.df[self.df_complete.columns[self.categorical_features[i]]].astype('category')
                categorical_features = pd.concat([categorical_features, self.df[self.df_complete.columns[self.categorical_features[i]]].cat.codes], axis=1)
        if not (len(categorical_features) == 0):
            mi = mutual_info_classif(categorical_features, self.df[self.class_label])
        counter = 0
        for column_nr in self.categorical_features:
            feature = self.column_names_list[column_nr]
            if feature not in self.drop_cols:
                self.json_data["info"]["analyzedFeatures"].append(feature)
                self.json_data["features"]["categoricalFeatures"][feature] = {}
                # Calculate missing values
                self.json_data["features"]["categoricalFeatures"][feature]['missingValues'] = self.miss_value[feature]
                # Identify levels
                (index_list, val_list, num_levels) = self.freq_counts(self.df, feature)
                levels = {}
                # Mongodb does not accept key name with dots.
                for i in range(len(val_list)):
                    if "." in str(index_list[i]):
                        index_list[i] = str(index_list[i]).replace(".", "")
                    levels[str(index_list[i])] = str(val_list[i])
                self.json_data["features"]["categoricalFeatures"][feature]['nrLevels'] = num_levels
                self.json_data["features"]["categoricalFeatures"][feature]['levels'] = levels
                # Calculate imbalance
                imbalance = self.imbalance_test(self.df, feature)
                self.json_data["features"]["categoricalFeatures"][feature]['imbalance'] = imbalance
                # Assign the mutual information for the feature
                self.json_data["features"]["categoricalFeatures"][feature]['mutualInfo'] = mi[counter]
                counter = counter + 1
                # Calculate correlation between selected feature and target feature.
                #(pval, chisq_correlated) = self.chisq_correlated_cal(self.csv_data, feature)
                #self.json_data["features"]["categoricalFeatures"][feature]['correlation'] = {}
                #self.json_data["features"]["categoricalFeatures"][feature]['correlation']['pVal'] = pval
                #self.json_data["features"]["categoricalFeatures"][feature]['correlation']['chisqCorrelated'] = chisq_correlated
                # Implement the monotonous filtering
                self.json_data["features"]["categoricalFeatures"][feature]['monotonousFiltering'] = self.monotonous_filtering_categorical(self.df, feature)
            else:
                self.json_data["info"]["discardedFeatures"].append(feature)
                print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")

    def analyse_unstructured_features(self):
        print("Analysing text features")
        self.json_data["features"]["unstructuredFeatures"] = {}
        #features = self.unstructured_features
        for column_nr in self.unstructured_features:
            feature = self.column_names_list[column_nr]
            if feature not in self.drop_cols:
                self.json_data["info"]["analyzedFeatures"].append(feature)
                self.json_data["features"]["unstructuredFeatures"][feature] = {}
                # Calculate missing values
                self.json_data["features"]["unstructuredFeatures"][feature]['missingValues'] = self.miss_value[feature]
                (vocab_size, relative_vocab, vocab_concentration, entropy, min_vocab, max_vocab) = self.text_statistics(self.df, feature)
                self.json_data["features"]["unstructuredFeatures"][feature]["vocabSize"] = vocab_size
                self.json_data["features"]["unstructuredFeatures"][feature]["relativeVocab"] = relative_vocab
                self.json_data["features"]["unstructuredFeatures"][feature]["vocabConcentration"] = vocab_concentration
                self.json_data["features"]["unstructuredFeatures"][feature]["entropy"] = entropy
                self.json_data["features"]["unstructuredFeatures"][feature]["minVocab"] = min_vocab
                self.json_data["features"]["unstructuredFeatures"][feature]["maxVocab"] = max_vocab

            else:
                self.json_data["info"]["discardedFeatures"].append(feature)
                print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")

    def datetime_features_computations(self, df, col_name):
        feature = df[col_name]
        sorted_feature = feature.sort_values(ascending=True, ignore_index=True)
        difference_dates = sorted_feature.diff()
        min_value = difference_dates.min()
        max_value = difference_dates.max()
        median_value = difference_dates.median()
        mean_value = difference_dates.mean()


        for i, date in enumerate(sorted_feature):
            sorted_feature[i] = datetime.datetime.fromtimestamp(sorted_feature[i])

        daypart_frequencies = np.zeros(3)
        month_frequencies = np.zeros(12)
        weekday_frequencies = np.zeros(7)
        hour_frequencies = np.zeros(24)
        for i in sorted_feature:
            if i.time().hour > 3 and i.time().hour <= 12:
                if i.time().hour == 12:
                    if i.time().minute == 0:
                        daypart_frequencies[0] += 1
                else:
                    daypart_frequencies[0] += 1
            elif i.time().hour >= 12 and i.time().hour <= 20:
                if i.time().hour == 12:
                    if not i.time().minute == 0:
                        daypart_frequencies[1] += 1
                elif i.time().hour == 20:
                    if i.time().minute == 0:
                        daypart_frequencies[1] += 1
                else:
                    daypart_frequencies[1] += 1
            else:
                daypart_frequencies[2] += 1

            month_frequencies[(i.date().month)-1] +=1
            weekday_frequencies[i.weekday()] +=1
            hour_frequencies[(i.time().hour)-1] +=1

        return min_value, max_value, mean_value, median_value, daypart_frequencies, month_frequencies, weekday_frequencies, hour_frequencies

    def analyse_datetime_features(self):
        print("Analysing datetime features")
        self.json_data["features"]["datetimeFeatures"] = {}
        dayparts = ['daypartMorning','daypartAfternoon','daypartEvening']
        months = ['monthJanuary','monthFebruary','monthMarch','monthApril','monthMay','monthJune','monthJuly','monthAugust','monthSeptember','monthOctober','monthNovmber','monthDecember']
        days = ['weekMonday','weekTuesday','weekWednesday','weekThursday','weekFriday','weekSaturday','weekSunday']
        hours = ['hour0','hour1','hour2','hour3','hour4','hour5','hour6','hour7','hour8','hour9','hour10','hour11','hour12','hour13','hour14','hour15','hour16','hour17','hour18','hour19','hour20','hour21','hour22','hour23',]
        for column_nr in self.datetime_features:
            feature = self.column_names_list[column_nr]
            if feature not in self.drop_cols:
                self.json_data["info"]["analyzedFeatures"].append(feature)
                self.json_data["features"]["datetimeFeatures"][feature] = {}
                # Calculate missing values
                self.json_data["features"]["datetimeFeatures"][feature]['missingValues'] = self.miss_value[feature]
                min_value, max_value, mean_value, median_value, daypart_frequencies, month_frequencies, weekday_frequencies, hour_frequencies = self.datetime_features_computations(self.df, feature)
                self.json_data["features"]["datetimeFeatures"][feature]['minDelta'] = min_value
                self.json_data["features"]["datetimeFeatures"][feature]['maxDelta'] = max_value
                self.json_data["features"]["datetimeFeatures"][feature]['meanDelta'] = mean_value
                self.json_data["features"]["datetimeFeatures"][feature]['medianDelta'] = median_value
                for i, value in enumerate(daypart_frequencies):
                    self.json_data["features"]["datetimeFeatures"][feature][
                        dayparts[i]] = value
                for i,value in enumerate(month_frequencies):
                    self.json_data["features"]["datetimeFeatures"][feature][
                        months[i]] = value
                for i,value in enumerate(weekday_frequencies):
                    self.json_data["features"]["datetimeFeatures"][feature][
                        days[i]] = value
                for i,value in enumerate(hour_frequencies):
                    self.json_data["features"]["datetimeFeatures"][feature][
                        hours[i]] = value

            else:
                self.json_data["info"]["discardedFeatures"].append(feature)
                print(feature + " is dropped for having missing values more than 1/4 the whole size of the dataset")

    @staticmethod
    def _convert_numpy_datatypes(json_data):
        return json.loads(json.dumps(json_data, default=DataProfiler.datatype_converter))

    # Main function which invokes all the other functions
    def analyse_dataset(self, mode: ReadMode, feature_annotation_list, dataset_path=None, dataset_string=None, dataset_df=None):
        print("Analysing Dataset")
        start = time.time()
        processing_status = self.process_pandas_df(mode, dataset_path, dataset_string, dataset_df)
        if not "processing success" in processing_status:
            error_message = "Please recheck target class label"
            return {}, error_message
        parse_feature_status = self.process_feature_annotation_list(feature_annotation_list)
        if not "parsing success" in parse_feature_status:
            print("Parsing Failed")
            error_message = "Please recheck feature type of the feature: " + parse_feature_status
            return {}, error_message
        self.calculate_ratios()
        analysis_status = self.analyse_numerical_features()
        if not "analysis success" in analysis_status:
            print("Analysis Failed")
            error_message = "Please recheck feature type of the feature: " + analysis_status
            return {}, error_message
        self.analyse_categorical_features()
        self.analyse_unstructured_features()
        self.analyse_datetime_features()
        stop = time.time()
        analysis_time = stop - start
        print(analysis_time)
        self.json_data["info"]["analysisTime"] = analysis_time
        #print(self.json_data["info"]["analysisTime"])

        return DataProfiler._convert_numpy_datatypes(self.json_data)


if __name__ == '__main__':
    # Command line arguments
    mode = int(sys.argv[1])
    dataset_path = ''
    base64_string = ''
    if mode == 1:
        dataset_path = sys.argv[2]
    elif mode == 2:
        base64_string = sys.argv[2]
    else:
        print("Invalid argument for Mode. Please try again with valid input")
        print("Accepted values : 1 or 2")
        exit(1)
    dataset_name = sys.argv[3]
    # print(dataset_name)
    target_label = sys.argv[4]
    # print(target_label)
    target_feature_type = sys.argv[5]
    feature_annotation_list = sys.argv[6]
    # print(feature_annotation_list)
    data_profiler = DataProfiler(dataset_name, target_label, target_feature_type)

    if mode == ReadMode.READ_CSV_FROM_FILE:
        data_profiler.analyse_dataset(mode, feature_annotation_list, dataset_path=dataset_path)
    elif mode == ReadMode.READ_CSV_FROM_BASE64:
        data_profiler.analyse_dataset(mode, feature_annotation_list, dataset_string=base64_string)
    else:
        print("Invalid argument for Mode. Please try again with valid input")
        print("Accepted values : 1 or 2")
        exit(1)
