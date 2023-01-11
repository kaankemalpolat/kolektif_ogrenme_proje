import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier


## 100 vs 10x10

# return bagging modeli, ensemble of ensemble modeli, df, df_diff
def baggingClassifier_100vs10x10(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        bagging = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        bagging_acc = bagging.fit(X_train, y_train).score(X_test, y_test)

        clf1 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf2 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf3 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf4 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf5 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf6 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf7 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf8 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf9 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf10 = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))

        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([bagging_acc,ens_acc])
    df = pd.DataFrame(result,columns=['bagging','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    bagging_vs_ens = [(df['bagging']<df['ens']).sum(),(df['bagging']==df['ens']).sum(),(df['bagging']>df['ens']).sum()]
    diff = df['ens']-df['bagging']

    return bagging, eclf, bagging_vs_ens, df, diff

# return ada modeli, ensemble of ensemble modeli, df, df_diff
def adaBoostClassifier_100vs10x10(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        ada = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        ada_acc = ada.fit(X_train, y_train).score(X_test, y_test)

        clf1 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf2 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf3 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf4 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf5 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf6 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf7 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf8 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf9 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf10 = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))

        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([ada_acc,ens_acc])
    df = pd.DataFrame(result,columns=['ada','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    ada_vs_ens = [(df['ada']<df['ens']).sum(),(df['ada']==df['ens']).sum(),(df['ada']>df['ens']).sum()]
    diff = df['ens']-df['ada']

    return ada, eclf, ada_vs_ens, df, diff


# return ada modeli, ensemble of ensemble modeli, df, df_diff
def randomSubspaceClassifier_100vs10x10(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        rnd_subspace = BaggingClassifier(bootstrap=False,n_estimators=100,random_state=random.randint(1,10000))
        rnd_subspace_acc = rnd_subspace.fit(X_train, y_train).score(X_test, y_test)

        clf1 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf2 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf3 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf4 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf5 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf6 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf7 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf8 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf9 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf10 = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))


        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([rnd_subspace_acc,ens_acc])
    df = pd.DataFrame(result,columns=['rnd_subspace','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    rnd_subspace_vs_ens = [(df['rnd_subspace']<df['ens']).sum(),(df['rnd_subspace']==df['ens']).sum(),(df['rnd_subspace']>df['ens']).sum()]
    diff = df['ens']-df['rnd_subspace']

    return rnd_subspace, eclf, rnd_subspace_vs_ens, df, diff


# return rf modeli, ensemble of ensemble modeli, df, df_diff
def randomForestClassifier_100vs10x10(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        rf = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        rf_acc = rf.fit(X_train, y_train).score(X_test, y_test)

        clf1 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf2 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf3 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf4 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf5 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf6 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf7 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf8 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf9 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        clf10 = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))

        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([rf_acc,ens_acc])
    df = pd.DataFrame(result,columns=['rf','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    rf_vs_ens = [(df['rf']<df['ens']).sum(),(df['rf']==df['ens']).sum(),(df['rf']>df['ens']).sum()]
    diff = df['ens']-df['rf']

    return rf, eclf, rf_vs_ens, df, diff

def extraTreesClassifier_100vs10x10(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        extra = ExtraTreesClassifier(n_estimators=100,random_state=random.randint(1,10000))
        extra_acc = extra.fit(X_train, y_train).score(X_test, y_test)

        clf1 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf2 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf3 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf4 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf5 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf6 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf7 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf8 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf9 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        clf10 = ExtraTreesClassifier(n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))


        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([extra_acc,ens_acc])
    df = pd.DataFrame(result,columns=['extra','ens'])

    # ensemble of ensemble için wind/draw/loss
    extra_vs_ens = [(df['extra']<df['ens']).sum(),(df['extra']==df['ens']).sum(),(df['extra']>df['ens']).sum()]
    diff = df['ens']-df['extra']

    return extra, eclf, extra_vs_ens, df, diff

# ====================================================================================================================

## 100 vs 10x100

# return bagging modeli, ensemble of ensemble modeli, df, df_diff
def baggingClassifier_100vs10x100(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        bagging = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        bagging_acc = bagging.fit(X_train, y_train).score(X_test, y_test)

        clf1 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf2 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf3 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf4 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf5 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf6 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf7 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf8 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf9 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf10 = BaggingClassifier(n_estimators=100,random_state=random.randint(1,10000))

        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([bagging_acc,ens_acc])
    df = pd.DataFrame(result,columns=['bagging','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    bagging_vs_ens = [(df['bagging']<df['ens']).sum(),(df['bagging']==df['ens']).sum(),(df['bagging']>df['ens']).sum()]
    diff = df['ens']-df['bagging']

    return bagging, eclf, bagging_vs_ens, df, diff


# return ada modeli, ensemble of ensemble modeli, df, df_diff
def adaBoostClassifier_100vs10x100(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        ada = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        ada_acc = ada.fit(X_train, y_train).score(X_test, y_test)

        clf1 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf2 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf3 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf4 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf5 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf6 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf7 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf8 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf9 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf10 = AdaBoostClassifier(n_estimators=100,random_state=random.randint(1,10000))

        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([ada_acc,ens_acc])
    df = pd.DataFrame(result,columns=['ada','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    ada_vs_ens = [(df['ada']<df['ens']).sum(),(df['ada']==df['ens']).sum(),(df['ada']>df['ens']).sum()]
    diff = df['ens']-df['ada']

    return ada, eclf, ada_vs_ens, df, diff
        
    
    # return ada modeli, ensemble of ensemble modeli, df, df_diff
def randomSubspaceClassifier_100vs10x100(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        rnd_subspace = BaggingClassifier(bootstrap=False,n_estimators=100,random_state=random.randint(1,10000))
        rnd_subspace_acc = rnd_subspace.fit(X_train, y_train).score(X_test, y_test)

        clf1 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf2 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf3 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf4 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf5 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf6 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf7 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf8 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf9 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf10 = BaggingClassifier(bootstrap=False,n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))


        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([rnd_subspace_acc,ens_acc])
    df = pd.DataFrame(result,columns=['rnd_subspace','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    rnd_subspace_vs_ens = [(df['rnd_subspace']<df['ens']).sum(),(df['rnd_subspace']==df['ens']).sum(),(df['rnd_subspace']>df['ens']).sum()]
    diff = df['ens']-df['rnd_subspace']

    return rnd_subspace, eclf, rnd_subspace_vs_ens, df, diff

    
# return rf modeli, ensemble of ensemble modeli, df, df_diff
def randomForestClassifier_100vs10x100(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        rf = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        rf_acc = rf.fit(X_train, y_train).score(X_test, y_test)

        clf1 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf2 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf3 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf4 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf5 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf6 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf7 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf8 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf9 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))
        clf10 = RandomForestClassifier(n_estimators=100,random_state=random.randint(1,10000))

        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([rf_acc,ens_acc])
    df = pd.DataFrame(result,columns=['rf','ens'])
        
    # ensemble of ensemble için wind/draw/loss
    rf_vs_ens = [(df['rf']<df['ens']).sum(),(df['rf']==df['ens']).sum(),(df['rf']>df['ens']).sum()]
    diff = df['ens']-df['rf']

    return rf, eclf, rf_vs_ens, df, diff

def extraTreesClassifier_100vs10x100(X_train, X_test, y_train, y_test):
    result = []
    for i in range(100):
        extra = ExtraTreesClassifier(n_estimators=100,random_state=random.randint(1,10000))
        extra_acc = extra.fit(X_train, y_train).score(X_test, y_test)

        clf1 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf2 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf3 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf4 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf5 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf6 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf7 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf8 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf9 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))
        clf10 = ExtraTreesClassifier(n_estimators=100,max_features=0.5,random_state=random.randint(1,10000))


        eclf = VotingClassifier(estimators=[('b1', clf1), ('b2', clf2), ('b3', clf3), ('b4', clf4), ('b5', clf5), ('b6', clf6), ('b7', clf7), ('b8', clf8), ('b9', clf9), ('b10', clf10),], voting='hard')
        ens_acc = eclf.fit(X_train, y_train).score(X_test, y_test)

        result.append([extra_acc,ens_acc])
    df = pd.DataFrame(result,columns=['extra','ens'])

    # ensemble of ensemble için wind/draw/loss
    extra_vs_ens = [(df['extra']<df['ens']).sum(),(df['extra']==df['ens']).sum(),(df['extra']>df['ens']).sum()]
    diff = df['ens']-df['extra']

    return extra, eclf, extra_vs_ens, df, diff

# ==========================================================================================================================
        
## plot

'''
inputs:
category_names = ['Ensemble of Ensemble', 'Draw','Ensemble']
results = {
    'Random Forest': [45,32,23],
    'GradBoosting': [0,100,0],
    'Bagging': [41,19,40],
    'Random Subspace': [44,21,35]
}
data_name = 'breast_cancer'
'''

def plot_ensemble_vs_eoe(results, category_names,data_name):

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['bwr'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'black' #if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0.5, -0.1),loc="lower center",fontsize='small')
    plt.title(data_name)
    return fig, ax


def error_kappa(model,X_test,y_test):
    error = []
    kappa = []
    for i in range(len(model.estimators_)-1):
        for j in range(i+1,len(model.estimators_)):
            m1 = model.estimators_[i]
            m2 = model.estimators_[i+1]

            m1_acc = m1.score(X_test,y_test)
            m2_acc = m2.score(X_test,y_test)

            error.append(1-(m1_acc+m2_acc)/2)

            m1_pred = m1.predict(X_test)
            m2_pred = m2.predict(X_test)

            kappa.append(cohen_kappa_score(m1_pred,m2_pred))
    return error, kappa

def plot_diff(df_diff,title_name):
    positive = df_diff.copy()
    positive[positive<0]=0
    
    negative = df_diff.copy()
    negative[negative>0]=0

    plt.bar(range(len(positive)),positive,color='blue')
    plt.bar(range(len(negative)),negative,color='red')
    plt.title(title_name)

    #====================================================================================================================

def single_vote_stack(X_train, X_test, y_train, y_test):
    bagging_scores = []
    ada_scores = []
    rs_scores = []
    rf_scores = []
    extra_scores = []

    vote_scores = []

    stack_bagging_scores = []
    stack_ada_scores = []
    stack_rs_scores = []
    stack_rf_scores = []
    stack_extra_scores = []

    score = []

    for i in range(10):
        bagging = BaggingClassifier(n_estimators=10,random_state=random.randint(1,10000))
        bagging_score = bagging.fit(X_train, y_train).score(X_test, y_test)
        bagging_scores.append(bagging_score)

        ada = AdaBoostClassifier(n_estimators=10,random_state=random.randint(1,10000))
        ada_score = ada.fit(X_train, y_train).score(X_test, y_test)
        ada_scores.append(ada_score)

        rs = BaggingClassifier(bootstrap=False,n_estimators=10,max_features=0.5,random_state=random.randint(1,10000))
        rs_score = rs.fit(X_train, y_train).score(X_test, y_test)
        rs_scores.append(rs_score)

        rf = RandomForestClassifier(n_estimators=10,random_state=random.randint(1,10000))
        rf_score = rf.fit(X_train, y_train).score(X_test, y_test)
        rf_scores.append(rf_score)

        extra = ExtraTreesClassifier(n_estimators=10,random_state=random.randint(1,10000))
        extra_score = extra.fit(X_train, y_train).score(X_test, y_test)
        extra_scores.append(extra_score)

        estimators = [('m1', bagging), ('m2', ada), ('m3', rs), ('m4', rf), ('m5', extra)]

        eclf = VotingClassifier(estimators=estimators, voting='hard')
        eclf_score = eclf.fit(X_train, y_train).score(X_test, y_test)
        vote_scores.append(eclf_score)


        stack_bagging_clf = StackingClassifier(estimators=estimators, final_estimator=BaggingClassifier(random_state=random.randint(1,10000)))
        stack_bagging_score = stack_bagging_clf.fit(X_train, y_train).score(X_test, y_test)
        stack_bagging_scores.append(stack_bagging_score)

        stack_ada_clf = StackingClassifier(estimators=estimators, final_estimator=AdaBoostClassifier(random_state=random.randint(1,10000)))
        stack_ada_score = stack_ada_clf.fit(X_train, y_train).score(X_test, y_test)
        stack_ada_scores.append(stack_ada_score)

        stack_rs_clf = StackingClassifier(estimators=estimators, final_estimator=BaggingClassifier(bootstrap=False,max_features=0.5,random_state=random.randint(1,10000)))
        stack_rs_score = stack_rs_clf.fit(X_train, y_train).score(X_test, y_test)
        stack_rs_scores.append(stack_rs_score)

        stack_rf_clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=random.randint(1,10000)))
        stack_rf_score = stack_rf_clf.fit(X_train, y_train).score(X_test, y_test)
        stack_rf_scores.append(stack_rf_score)

        stack_extra_clf = StackingClassifier(estimators=estimators, final_estimator=ExtraTreesClassifier(random_state=random.randint(1,10000)))
        stack_extra_score = stack_extra_clf.fit(X_train, y_train).score(X_test, y_test)
        stack_extra_scores.append(stack_extra_score)

    score.append(['Bagging',sum(bagging_scores)/len(bagging_scores)])
    score.append(['Adaboost',sum(ada_scores)/len(ada_scores)])
    score.append(['RS',sum(rs_scores)/len(rs_scores)])
    score.append(['RF',sum(rf_scores)/len(rf_scores)])
    score.append(['ExtraRandomTree',sum(extra_scores)/len(extra_scores)])

    score.append(['Voting',sum(bagging_scores)/len(vote_scores)])

    score.append(['Stack_Bagging',sum(stack_bagging_scores)/len(stack_bagging_scores)])
    score.append(['Stack_Adaboost',sum(stack_ada_scores)/len(stack_ada_scores)])
    score.append(['Stack_RS',sum(stack_rs_scores)/len(stack_rs_scores)])
    score.append(['Stack_RF',sum(stack_rf_scores)/len(stack_rf_scores)])
    score.append(['Stack_ExtraRandomTree',sum(stack_extra_scores)/len(stack_extra_scores)])

    df = pd.DataFrame(score)
    #df.sort_values(by=[1],ascending=False)
    return df
