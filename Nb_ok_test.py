# coding: utf-8

# NB

# get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import preprocessing
from django.views.generic import TemplateView
import os
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from plotly_pic import compare_facies_plot


class nb_algorighm(TemplateView):
    def nb_algorighm1(self):
        set_option("display.max_rows", 10)
        pd.options.mode.chained_assignment = None

        file_name = 'EMDdata.csv'
        training_data = pd.read_csv(os.path.join('data\svm_smp\input\cvs', file_name), sep='\t')
        # training_data = pd.read_csv('data/EMDdata.csv', sep='\t')
        print(training_data)


        print(os.path.join('data\svm_smp\input\cvs', file_name))
        # training_data = pd.read_csv('datawmTrees.csv', sep='\t')

        # training_data

        training_data['Well Name'] = training_data['Well Name'].astype('category')
        training_data['Formation'] = training_data['Formation'].astype('category')
        training_data['Well Name'].unique()
        training_data = training_data.drop(['Formation'], axis=1)
        training_data = training_data.drop(['Unnamed: 0'], axis=1)

        blind = training_data[training_data['Well Name'] == 'KIMZEY A']
        training_data = training_data[training_data['Well Name'] != 'KIMZEY A']

        # code is borrowed from @kwinkunks
        ###
        facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
                         '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

        facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                         'WS', 'D', 'PS', 'BS']
        # facies_color_map is a dictionary that maps facies labels
        # to their respective colors
        facies_color_map = {}
        for ind, label in enumerate(facies_labels):
            facies_color_map[label] = facies_colors[ind]

        def label_facies(row, labels):
            return labels[row['Facies'] - 1]

        training_data.loc[:, 'FaciesLabels'] = training_data.apply(
            # lambda row: nb_algorighm.label_facies(row, nb_algorighm.facies_labels), axis=1)
            lambda row: label_facies(row, facies_labels), axis = 1)

        correct_facies_labels = training_data['Facies'].values

        feature_vectors = training_data.drop(['Well Name', 'Depth', 'Facies', 'FaciesLabels'], axis=1)

        scaled_features = (feature_vectors - np.mean(feature_vectors)) / np.std(feature_vectors)

        X_train, X_test, y_train, y_test = train_test_split(scaled_features, correct_facies_labels, test_size=0.2)

        # clf = GaussianNB()

        correct_facies_labels = training_data['Facies'].values

        feature_vectors = training_data.drop(['Well Name', 'Depth', 'Facies', 'FaciesLabels'], axis=1)

        scaled_features = (feature_vectors - np.mean(feature_vectors)) / np.std(feature_vectors)

        # X_train = scaled_features
        # y_train = correct_facies_labels

        y_blind = blind['Facies'].values

        well_features = blind.drop(['Well Name', 'Depth', 'Facies'], axis=1)

        X_blind = (well_features - np.mean(well_features)) / np.std(well_features)

        clf = GaussianNB()

        clf.fit(X_train, y_train)

        res = {}

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_blind)
        c = 0
        for i in range(len(y_pred)):
            if y_blind[i] == y_pred[i]:
                c = c + 1
        res = c / float(len(y_pred))

        # res

        y_pred = clf.predict(X_blind)
        y_pred = y_pred.astype(int)
        blind['Prediction'] = y_pred


        compare_facies_plot(blind, 'Prediction', facies_colors)

        Precision = {}
        Recall = {}

        for j in range(1, 10):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(1, len(y_pred)):
                if ((y_blind[i] == j) and (y_pred[i] == j)):
                    TP = TP + 1
                if ((y_blind[i] != j) and (y_pred[i] == j)):
                    FP = FP + 1
                if ((y_blind[i] == j) and (y_pred[i] != j)):
                    FN = FN + 1
                if ((y_blind[i] != j) and (y_pred[i] != j)):
                    TN = TN + 1
            if (TP + FP == 0):
                Precision[j] = 0
            else:
                Precision[j] = float(TP) / (TP + FP)
            if (TP + FN == 0):
                Recall[j] = 0
            else:
                Recall[j] = float(TP) / (TP + FN)

        X = np.arange(9)
        res = {'Precision': Precision, 'Recall': Recall}

        Res = pd.DataFrame(res)
        Res['Facies'] = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
        plt.bar(X - 0.2, Res['Precision'], color='lightblue', width=0.4, label='Precision')
        plt.bar(X + 0.2, Res['Recall'], color='brown', width=0.4, label='Recall')
        plt.title('Precision & Recall')
        plt.xticks(X, Res['Facies'])
        plt.legend()
        file_name = 'pic1.png'
        file_pic1 =os.path.join('data\svm_smp\output\plots', file_name)
        print(file_pic1)
        plt.savefig(file_pic1)

        plt.show()


        # Precision

        # Recall

        f = {}
        for i in range(1, 10):
            if ((Precision[i] == 0) or (Recall[i] == 0)):
                f[i] = 0
            else:
                f[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])

        f = {'F': f}
        F = pd.DataFrame(f)
        X = np.arange(9)
        F['Facies'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.bar(X, F['F'], color='brown', width=0.5, label='F')
        plt.title('F')
        plt.xticks(X, Res['Facies'])
        file_name = 'pic2.png'
        file_pic2 =os.path.join('data\svm_smp\output\plots', file_name)
        print(file_pic2)
        plt.savefig(file_pic2)
        plt.show()



        # F

        # postprocessing
        n = 2

        for i in range(n - 1, len(y_pred) - n):
            check = 0
            for j in range(n + 1):
                if ((y_pred[i] != 1) and ((y_pred[i] != y_pred[i - np.asarray(range(n + 1)) + j]).any())):
                    check = check + 1
            if (check == n + 1):
                am = [0, 0]
                k1 = 1
                while (y_pred[i - 1] == y_pred[i - k1]):
                    am[0] = am[0] + 1
                    if ((i - k1) == 0):
                        break
                    k1 = k1 + 1
                k2 = n - 1
                while (y_pred[i + 1] == y_pred[i + k2]):
                    am[1] = am[1] + 1
                    if (i + k2 != (len(y_pred) - n)):
                        k2 = k2 + 1
                    else:
                        break
                if (am[0] == max(am)):
                    y_pred[i] = y_pred[i - 1]
                else:
                    y_pred[i] = y_pred[i + 1]

        p = 0

        for i in range(1, len(y_pred)):
            if y_pred[i] == y_blind[i]:
                p = p + 1
        accuracy = p / float(len(y_pred))

        print('Facies classification accuracy（相分类精度） = %f' % accuracy)

        blind['Prediction'] = y_pred
        compare_facies_plot(blind, 'Prediction', facies_colors)

        Precision = {}
        Recall = {}

        for j in range(1, 10):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(1, len(y_pred)):
                if ((y_blind[i] == j) and (y_pred[i] == j)):
                    TP = TP + 1
                if ((y_blind[i] != j) and (y_pred[i] == j)):
                    FP = FP + 1
                if ((y_blind[i] == j) and (y_pred[i] != j)):
                    FN = FN + 1
                if ((y_blind[i] != j) and (y_pred[i] != j)):
                    TN = TN + 1
            if (TP + FP == 0):
                Precision[j] = 0
            else:
                Precision[j] = float(TP) / (TP + FP)
            if (TP + FN == 0):
                Recall[j] = 0
            else:
                Recall[j] = float(TP) / (TP + FN)

        # Precision

        # Recall

        f = {}
        for i in range(1, 10):
            if ((Precision[i] == 0) or (Recall[i] == 0)):
                f[i] = 0
            else:
                f[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])

        def sort_values(self, by):
            pass
        num =[file_pic1, file_pic2, accuracy]
        return num

        # print(nb_algorighm())

# print(nb_algorighm().nb_algorighm1())

