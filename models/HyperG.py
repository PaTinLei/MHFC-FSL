import math
import random

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import scipy.sparse as sparse

from sklearn.preprocessing import normalize
import time

class HyperG(object):

    def __init__(self, classifier='lr', num_class=None, step=5, max_iter='auto',
                 reduce='pca', d=32,rate = 0.5,norm='l2'):
        self.step = step
        self.dim = d
        self.max_iter = max_iter
        self.num_class = num_class
        self.initial_norm(norm)
        self.initial_classifier_1(classifier)
        self.initial_classifier_2(classifier)
        self.initial_classifier_multi(classifier)
        self.rate = rate
    def init_label_matrix(self, y):
        """
        :param y: numpy array, shape = (n_nodes,) -1 for the unlabeled data, 0,1,2.. for the labeled data
        :return:
        """
        y = y.reshape(-1)
        labels = list(np.unique(y))

        if -1 in labels:
            labels.remove(-1)

        n_nodes = y.shape[0]
        Y = np.ones((n_nodes, len(labels))) * (1/len(labels))
        for idx, label in enumerate(labels):
            Y[np.where(y == label), :] = 0
            Y[np.where(y == label), idx] = 1

        return Y

    def fit(self, X_task, y):
        self.support_X_task = self.norm(X_task)
        self.support_X_task_no_norm = X_task
        self.support_y = y

    def predict(self, X1, unlabel_X_task=None, show_detail=False, query_y=None,eta=None):
        self.eta = eta
        support_X_task, support_y = self.support_X_task, self.support_y
        way, num_support = self.num_class, len(support_X_task)
        query_X_task = X1#self.norm(X1)   #unlabel_X
        support_X_task_no_norm = self.support_X_task_no_norm
        if unlabel_X_task is None:
            unlabel_X_task = query_X_task
            unlabel_X_task_no_norm = query_X_task
        else:
            unlabel_X_task_no_norm = unlabel_X_task
            unlabel_X_task = unlabel_X_task
        num_unlabel = unlabel_X_task.shape[0]

        assert self.support_X_task is not None
        self.embeddings_task = np.concatenate([support_X_task, unlabel_X_task])
        self.embeddings_task_no_norm = np.concatenate([support_X_task_no_norm, unlabel_X_task_no_norm])

        if self.max_iter == 'auto':
            self.max_iter = num_support + num_unlabel
        elif self.max_iter == 'fix':
            self.max_iter = math.ceil(num_unlabel/self.step)
        else:
            assert float(self.max_iter).is_integer()

        support_set_task = np.arange(num_support).tolist()

        support_X_multi = self.multi_task_soft(self.support_X_task, self.support_y)
        support_X_multi_norm = self.norm(support_X_multi)
        self.classifier_multi.fit(support_X_multi_norm[support_set_task], self.support_y)
 
        self.classifier_1.fit(self.norm(support_X_multi[support_set_task][:,:self.dim]), self.support_y)
        self.classifier_2.fit(self.norm(support_X_multi[support_set_task][:,self.dim:]), self.support_y)


        if show_detail:
            acc_list = []

            query_X_task_predict = np.concatenate((query_X_task[:,:self.dim]*self.omega[0],query_X_task[:,self.dim:]*self.omega[1]), axis=1)
            query_X_task_predict = self.norm(query_X_task_predict)
            task1 = self.classifier_1.predict(self.norm(query_X_task[:,:self.dim]*self.omega[0]))
            task2 = self.classifier_2.predict(self.norm(query_X_task[:,self.dim:]*self.omega[1]))

            soft_Y_task = self.classifier_multi.predict_proba(query_X_task_predict)
            inductive_predicts_task = self.classifier_multi.predict(query_X_task_predict)

            acc_list.append(accuracy_score(query_y, inductive_predicts_task))

        for iter in range(self.max_iter):
            unlabel_X_task_predict = np.concatenate((unlabel_X_task[:,:self.dim]*self.omega[0],unlabel_X_task[:,self.dim:]*self.omega[1]), axis=1)
            unlabel_X_task_predict = self.norm(unlabel_X_task_predict)
            soft_Y_task = self.classifier_multi.predict_proba(unlabel_X_task_predict)
            pseudo_y_task = self.classifier_multi.predict(unlabel_X_task_predict)
			
            y_task = np.concatenate([support_y, pseudo_y_task])
            Y_task = self.label2onehot(y_task, way)

            support_set_task = self.expand(iter,  way, num_support, soft_Y_task, pseudo_y_task)

            y_task = np.argmax(Y_task, axis=1)
            support_X_multi = self.multi_task_soft(self.embeddings_task[support_set_task], y_task[support_set_task])
            support_X_multi_norm = self.norm(support_X_multi)
            self.classifier_multi.fit(support_X_multi_norm[support_set_task], y_task[support_set_task])
            query_X_task_predict = np.concatenate((query_X_task[:,:self.dim]*self.omega[0],query_X_task[:,self.dim:]*self.omega[1]), axis=1)
            query_X_task_predict = self.norm(query_X_task_predict)
            predicts_task = self.classifier_multi.predict(query_X_task_predict)

            self.classifier_1.fit(self.norm(support_X_multi[support_set_task][:,:self.dim]), y_task[support_set_task])
            self.classifier_2.fit(self.norm(support_X_multi[support_set_task][:,self.dim:]), y_task[support_set_task])

            task1 = self.classifier_1.predict(self.norm(query_X_task[:,:self.dim]*self.omega[0]))
            task2 = self.classifier_2.predict(self.norm(query_X_task[:,self.dim:]*self.omega[1]))

            if show_detail:
                acc_list.append(accuracy_score(query_y, predicts_task))


            if(iter==((num_unlabel//5)+5)):
                break
        
        return acc_list


    def multi_task_soft(self, support_X,support_y):

        x_task, y_task, x_softmax, y_softmax = support_X[:,:self.dim], support_y, support_X[:,self.dim:], support_y
        n_hg = 2
        eta = self.eta
        loss = np.zeros(n_hg)

        self.classifier_1.fit(x_task, y_task)
        self.classifier_2.fit(x_softmax, y_softmax)

        wx_task = self.classifier_1.decision_function(x_task)
        wx_softmax = self.classifier_2.decision_function(x_softmax)

        Y_task = self.init_label_matrix(y_task)
        Y_softmax = self.init_label_matrix(y_softmax)

        Y_XW_task = Y_task - self.sigmoid(wx_task)
        Y_XW_softmax = Y_softmax - self.sigmoid(wx_softmax)

        x_task_inv = np.linalg.pinv(x_task)
        x_softmax_inv = np.linalg.pinv(x_softmax)

        w_task = x_task_inv.dot(wx_task)
        w_softmax = x_softmax_inv.dot(wx_softmax)

        loss_task = np.trace(Y_XW_task.T.dot(Y_XW_task))+np.trace(w_task.T.dot(w_task))
        loss_softmax = np.trace(Y_XW_softmax.T.dot(Y_XW_softmax))+np.trace(w_softmax.T.dot(w_softmax))
 
        loss[0] = loss_task
        loss[1] = loss_softmax

        loss = self.normalization(loss)

        self.omega = self.EProjSimplex_v2(loss, 1, eta)

        test_embeddings_task = np.concatenate((self.embeddings_task_no_norm[:,:self.dim]*self.omega[0],self.embeddings_task_no_norm[:,self.dim:]*self.omega[1]), axis=1)
        
        return test_embeddings_task

    def expand(self, iter,way, num_support, soft_Y, pseudo_y):
        iter = iter + 1

        soft_Y_max = np.max(soft_Y,axis=1)
        init_support_set = np.arange(num_support).tolist()

        way_index = []
        way_index_lenth = np.zeros(way)
        for i in range(way):
            pseudo_y_where = np.where(pseudo_y == i)[0]
            way_index.append(pseudo_y_where)
            way_index_lenth[i] = len(pseudo_y_where)

        way_index_lenth_min = int(np.min(way_index_lenth))
        if iter >= way_index_lenth_min:
            iter = way_index_lenth_min
        for j in range(iter):
            for i in range(way):
                if(way_index_lenth[i] == 0):
                   continue
                soft_Y_max_way_index = soft_Y_max[way_index[i]]
                Z = zip(soft_Y_max_way_index,way_index[i])
                Z_reverse = sorted(Z,reverse=True)
                A_new,B_new = zip(*Z_reverse)
				
                init_support_set.append(num_support+B_new[j])

        return init_support_set

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def normalization(self,data):
        _range = data/math.sqrt(np.sum(data*data))
        return _range

    def EProjSimplex_v2(self,v, k, eta):
        ft = 1
        n = len(v)
        x = np.array([0,0])
        v0 = (-v + (2*eta+np.sum(v))/n)
        vmin = min(v0)

        if(vmin<0):
            f = 1
            lambda_m = 0
            while (abs(f) > 10^-10):
                v1 = (v0 - lambda_m)
                posidx = v1>0
                posidx = posidx.astype(int)
                npos = np.sum(posidx)
                g = -npos
                f = np.sum(v1[posidx]) - 2*eta*k  
                lambda_m = lambda_m - f/g
                ft=ft+1
                if ft > 100:
                    x = np.maximum(v1,0)/(2*eta)
                    break
                x = np.maximum(v1,0)/(2*eta)
        else:
            x = v0/(2*eta);
        return x

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x

    def initial_classifier_1(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(C=10, gamma='auto', kernel='linear',probability=True)
        elif classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier_1 = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def initial_classifier_2(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier_2 = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def initial_classifier_multi(self, classifier):
        assert classifier in ['lr', 'svm']
        if classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.classifier_multi = LogisticRegression(
                C=10, multi_class='auto', solver='lbfgs', max_iter=1000)

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result
