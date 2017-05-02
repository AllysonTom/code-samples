from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn import preprocessing
from xgboost import XGBClassifier as xgb
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# hitters voting classifier using the optimal parameters
def voting_hitters(to_predict_hitters, hitter_predictions, x_hitters, xgb_hitters_params, rforest_hitters_params, 
                    logreg_hitters_params, qda_hitters_params):
    accuracy_list_voting_hitters = []
    accuracy_list_voting_hitters_AVG = []
    accuracy_list_voting_hitters_HR = []
    accuracy_list_voting_hitters_R = []
    accuracy_list_voting_hitters_RBI = []
    accuracy_list_voting_hitters_SB = []

    for i in xrange(10):
        j=0
        for col in to_predict_hitters:
            y = hitter_predictions[col].tolist()
            clf1 = xgb(**xgb_hitters_params[j][col])
            clf2 = RandomForestClassifier(**rforest_hitters_params[j][col])
            clf3 = linear_model.LogisticRegression(**logreg_hitters_params[j][col])
            clf4 = QuadraticDiscriminantAnalysis(**qda_hitters_params[j][col])
            #clf4 = svm.SVC(**svm_hitters_params[j][col])
            eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('qda', clf4)], voting='soft')
            
            scores = cross_val_score(eclf, x_hitters, y, cv=5, scoring='accuracy', n_jobs = -1)

            acc = scores.mean()
            accuracy_list_voting_hitters.append(acc)
            
            if col == 'correct_AVG':
                accuracy_list_voting_hitters_AVG.append(acc)
            elif col == 'correct_HR':
                accuracy_list_voting_hitters_HR.append(acc)
            elif col == 'correct_R':
                accuracy_list_voting_hitters_R.append(acc)
            elif col == 'correct_RBI':
                accuracy_list_voting_hitters_RBI.append(acc)
            elif col == 'correct_SB':
                accuracy_list_voting_hitters_SB.append(acc)
            j+=1
            
    print "%-15s" % 'overall average', np.mean(accuracy_list_voting_hitters)
    print "%-15s" % 'correct_AVG', np.mean(accuracy_list_voting_hitters_AVG)
    print "%-15s" % 'correct_HR', np.mean(accuracy_list_voting_hitters_HR)
    print "%-15s" % 'correct_R', np.mean(accuracy_list_voting_hitters_R)
    print "%-15s" % 'correct_RBI', np.mean(accuracy_list_voting_hitters_RBI)
    print "%-15s" % 'correct_SB', np.mean(accuracy_list_voting_hitters_SB)
def voting_pitchers(to_predict_pitchers, pitcher_predictions, x_pitchers, xgb_pitchers_params, rforest_pitchers_params, logreg_pitchers_params, 
                    svm_pitchers_params):
    # pitchers voting classifier using the optimal parameters

    accuracy_list_voting_pitchers = []
    accuracy_list_voting_pitchers_ERA = []
    accuracy_list_voting_pitchers_K = []
    accuracy_list_voting_pitchers_W = []
    accuracy_list_voting_pitchers_WHIP = []

    i=0
    col_list = ['correct_ERA', 'correct_K', 'correct_W', 'correct_WHIP']
    for col in col_list:
        svm_pitchers_params[i][col]['probability'] = True
        i+=1

    for i in xrange(10):
        j=0
        for col in to_predict_pitchers:
            y = pitcher_predictions[col].tolist()
            clf1 = xgb(**xgb_pitchers_params[j][col])
            clf2 = RandomForestClassifier(**rforest_pitchers_params[j][col])
            clf3 = linear_model.LogisticRegression(**logreg_pitchers_params[j][col])
            #clf5 = QuadraticDiscriminantAnalysis(**qda_pitchers_params[j][col])
            clf4 = svm.SVC(**svm_pitchers_params[j][col])
            eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)], voting='soft')
            #eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
            scores = cross_val_score(eclf, x_pitchers, y, cv=5, scoring='accuracy', n_jobs=-1)
            
            acc = scores.mean()
            accuracy_list_voting_pitchers.append(acc)
            
            if col == 'correct_ERA':
                accuracy_list_voting_pitchers_ERA.append(acc)
            elif col == 'correct_K':
                accuracy_list_voting_pitchers_K.append(acc)
            elif col == 'correct_W':
                accuracy_list_voting_pitchers_W.append(acc)
            elif col == 'correct_WHIP':
                accuracy_list_voting_pitchers_WHIP.append(acc)
            j+=1

    print "%-15s" % 'overall average', np.mean(accuracy_list_voting_pitchers)
    print "%-15s" % 'correct_ERA', np.mean(accuracy_list_voting_pitchers_ERA)
    print "%-15s" % 'correct_K', np.mean(accuracy_list_voting_pitchers_K)
    print "%-15s" % 'correct_W', np.mean(accuracy_list_voting_pitchers_W)
    print "%-15s" % 'correct_WHIP', np.mean(accuracy_list_voting_pitchers_WHIP)
