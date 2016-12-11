import pandas as pd
from datetime import datetime
import time
import numpy as np
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
import pymc3 as pm  
from scipy import optimize
from math import lgamma, factorial
from numpy import log
from statsmodels.formula.api import ols
from scipy.stats import beta
#
class diary():
    def __init__(self,fname):
        """
        This method will initialize the class with a file name
        """
        wts = np.array([0.5,0.4,0.1,0,-100,-10,10,20,30,15,0])
        self.dict = {33:0,34:1,35:2,65:3,66:4,67:5,68:6,69:7,70:8,71:9,72:10}
        f = open(fname,'r')
        rec=[]
        for line in f:
            index = line.split('\t')[0]+' '+line.split('\t')[1]
            time_tuple = time.strptime(index, "%m-%d-%Y %H:%M")
            timestamp = int(time.mktime(time_tuple))
            observation = int(line.split('\t')[2])
            numeric = float(line.split('\t')[3].rstrip('\n'))
     
            rec.append([index,timestamp,observation,numeric])
        self.data = pd.DataFrame(rec,columns=['time','timestamp','observation','numeric'])
        #
        self.states  = self.data[(self.data['observation']>=48) &(self.data['observation']<=64)]
        indices = zip(range(0,self.states.shape[0],2),range(1,self.states.shape[0],2))  ###1
        s2sp=[]
        for i,j in indices:
            s2sp.append([self.states.ix[self.states.index[i]][['timestamp','numeric']].tolist(),self.states.ix[self.states.index[j]][['timestamp','numeric']].tolist()])
        #
        self.trans = pd.DataFrame([[pre[0],pre[1],post[0],post[1]] for (pre,post) in s2sp],columns=['ts_s','s','ts_sp','sp'])
        #
        self.df=[]
        for id in self.trans.index:
            observect = np.zeros((11))
            start = self.trans.ix[id]['ts_s']
            stop = self.trans.ix[id]['ts_sp']
            s = self.trans.ix[id]['s']
            sp = self.trans.ix[id]['sp']
            #
            start_id = self.data[self.data['timestamp']==start].index.tolist()[0]
            stop_id = self.data[self.data['timestamp']==stop].index.tolist()[0]
            activity = [(r,list(self.data.ix[r][['observation','numeric']].values)) for r in range(start_id+1,stop_id,1)]
            #print((id,start,activity,stop))
            print(self.data.ix[start_id:stop_id])
            for ac in activity:
                observect[self.dict[ac[1][0]]] = ac[1][1]
            delta_t = format((stop-start)/3600,'.2f')
            record = [id,s,sp,float(delta_t)]
            action = np.dot(observect,wts)  # assign action
            print(action)
            record.append(action)
            print(record)
            record.extend(list(observect))
            self.df.append(record)
        self.glyc = pd.DataFrame(np.matrix(self.df) ,columns=['id','s','sp','delta_t','axt','33','34','35','65','66','67','68','69','70','71','72'])
       
        self.glyc['r'] = self.get_reward(self.glyc['sp'])
        #
        self.glyc['reg']=self.glyc['33']/self.glyc[['33','34','35']].sum(axis=1)
        self.glyc['nph']=self.glyc['34']/self.glyc[['33','34','35']].sum(axis=1)
        self.glyc['lente']=self.glyc['35']/self.glyc[['33','34','35']].sum(axis=1)
        pool_gamma = (self.glyc[['33','34','35']].sum(axis=0))+1
        pool_gamma=pool_gamma/pool_gamma.sum()
        lk=[]
        for i in self.glyc.index:
            X=self.glyc[['33','34','35']].ix[i,:].values
            lk.append(self.log_pmf(X,pool_gamma))
        self.glyc['lk']=lk
       
        self.glyc.fillna(0,inplace=True)
        # compute action class
        XX=self.glyc['axt'].reshape(self.glyc.shape[0],1)
        X, n,best_fit = self.get_GMM(XX)
        a = best_fit.predict(X)    # assign classification to action according to GMM
        self.glyc['a'] =a
        self.diabetic = pd.DataFrame()
        self.diabetic[['s','r','delta_t','sp','a']] = self.glyc[['s','r','delta_t','sp','a']]
        #self.diabetic['a']=a

    def get_reward(self,x):
        """
        This method will compute the reward for glucose result. 
        """
        Xs = np.array([np.power(x,2),x,1])
        cs = np.array([-1,215,-9750])
        return list(np.dot(Xs,cs))
        
    def get_GMM(self,X):
        """
        This method will assign a GMM mixture model to the diffecrent action choices of the patient. I will select 
        the best number of clusers and return the class the action belongs to. If the number of clustering is one; ie
        fit a normal gaussian it will assign action accoring to four set of quantiles.
        The number of clusters will be assigned using aic.
        return: class label, weights, means and covariance

        """
        groups = np.arange(1,11)   # assume there are at best 10 different action options
        #X = self.glyc['axt'].reshape(self.glyc.shape[0],1)
        fit = [GMM(n,n_iter=500).fit(X) for n in groups]
        fit_aic = [f.aic(X)   for f in fit]
        aic_idx = fit_aic.index(min(fit_aic))
        n_clusters = fit[aic_idx].n_components

        return X,n_clusters,fit[aic_idx]

    def show_GMM(self):
        """
        A method to plot the fit of the density and the observed distribution
        """
        X,n_clusters,best_fit = self.get_GMM(XX)
        xx = np.linspace(min(X)-5,max(X)+5,1000)
        xx=xx.reshape(xx.shape[0],1)
        dens = np.exp(best_fit.score(xx))
        #
        plt.hist(X,bins=10,normed=True,alpha=0.4)
        plt.plot(xx,dens,'-r')

        return n_clusters,dens

  

    def log_pmf(self,X,p):
        """
        A method to compute the probability mass function of a multinomial given observation X and probability vector p
        """
        n=sum(X)
        llklhd =  lgamma(n+1)- sum([lgamma(x+1) for x in X])+np.dot(X,np.array([log(P) for P in p]))
        return llklhd
        
    def simulate_response(self,model):
        """
        This method is desined to sample the sample space of different insulins with different plasma glucose levels
        and different time stampls. Ideally  we will look at 2 hours to limit the state space of options.
        The choice of insulins is in the clinically feasible range
        regular insulin 0-20 units
        NPH 0-50 units
        Lente 0-50 units
        
        """
        sim_rec=[]
        reg_max=20
        nph_max = 50
        lent_max = 50
        delta_t=2
        params = model.params
        for r in range(0,reg_max):
            for n in range(0,nph_max):
                for l in range(0,lent_max):
                    for s in range(55,300):
                        vars = np.array([1,s,delta_t,r,n,l])
                        expect = np.round(np.dot(vars,model.params))
                        #r = self.get_reward(expect)
                        sim_rec.append([s,r,n,l,expect])
        df = pd.DataFrame(sim_rec,columns=['s','reg','nph','lente','sp'])
        df.to_csv('sim_glyc.csv')
        return df 
            
    def get_godel(self,X):
        return np.exp(X['reg']*np.log(2)+X['nph']*log(3)+X['lente']*log(5))
   
    def get_pmf(self,X):
        """
        returns the pmf of observing a given insulin combination given a known probability
        """
        alpha = X.astype(int).sum(axis=0)
        pool_prob = alpha/alpha.sum()
        llk = []
        for i in X.index:
            x=X.ix[i,:]
            llk.append([self.log_pmf(x,pool_prob)])
        
        return pd.DataFrme(llk,columns=['llk'] )   


    def sim_solution(self,model):
        """
        This method will simulate a plasma serum s and 50 different actions
        compute the reward and the sp and choose the best action for s
        The action a is distributed dirichlet and it will set the best action as its best starting point for next time we arrive in state s
        Over time there will be different estimates for state s which are optimal .
        The idea is that over time we will converge to the optimal dirichlet parameters for action from state s:
        """
        # Intialize vector U size 1x300<-0. We will only use the range 55-300. Any plasma level below 55 is hypogoglycemia.
        iter = 4000   # number of iterations
        min_plasma =55
        max_plasma=300
        insulin_range = 30
        U = np.zeros((1,max_plasma))
        # set a dictionary of state:dirichlet
        action_mat = np.ones((max_plasma,3))    # initialize the dirichlet
        reward = np.zeros((max_plasma,))
        trace = {i:[]for i in range(max_plasma)}
        #
        n_dirichlet=100  # numebr of samples to draw.
        best={}  # a dictionary to hold the best response to s by choosing a and getting reward r
        #
        for i in range(iter):
            s = np.random.randint(min_plasma,max_plasma,1)  # draw and observation from s 55-300
            r = np.zeros((max_plasma,))
            # draw 30 dirichet 
            act = np.random.randint(0,insulin_range,(n_dirichlet,3))  # draw dirichlet , comment in future work.
            add = np.ones((n_dirichlet,3))*np.array([1,s,2])     #add the  intercept, 2h time and drawn plasm glucose
            expectation = np.dot(np.concatenate((add,act),axis=1),model.params) #  get sp 
            if np.any([expectation>=55]):  #  non negatives and no hypoglycemics
                #compute the reward
                exp_reward = self.get_reward(expectation)
                #which action had the best reward
                action_id = exp_reward.index(max(exp_reward))

                action_mat[s,:] += act[(np.where(exp_reward==max(exp_reward))[0])[0]]
    
                trace[s[0]].extend(list(act[(np.where(exp_reward==max(exp_reward))[0])[0]]))
                reward[s[0]] = max(exp_reward)

        return action_mat,reward,trace


    def process_trace(self,trace,upper=300, lower=55):
            """This method will reformat the trace and report the mean and sd for each category in alpha"""
            reshape={k:np.array(t[k]).reshape(int(len(t[k])/3.0),3) for k in t.keys()}
            trace_mean = {k:reshape[k].mean(axis=0) for k in reshape.keys()}
            trace_sd = {k:reshape[k].std(axis=0) for k in reshape.keys()} 
            return trace_mean,trace_sd

    def trace_solution(self,trace,model,upper=300,lower=55):
            """
            This method will report a trace of simulated results for a patient
            """
            plt.close('all')
            plasma = np.zeros((upper,1))
            trace_mean,trace_sd = self.process_trace(trace)
            for s in range(lower,upper):
                mu,sigma = trace_mean[s],trace_sd[s]
                print(s)
                #plasma[s] = np.dot(np.concatenate((np.array([1,s,2]),np.random.normal(mu,sigma+0.001,size=(3,))),axis=0),model.params)
                plasma[s] = np.dot(np.concatenate((np.array([1,2,3]),mu),axis=0),model.params)
            # see how well patient did:
            pt_plasma = np.dot(np.concatenate((np.ones((d.glyc.shape[0],1)), d.glyc[['s','delta_t','33','34','35']].values),axis=1),model.params)
            fig, (ax,ax1) = plt.subplots(1,2)
            ax.scatter(range(0,upper),plasma,s=0.75,c='blue',marker='*',label='algorithm')
            ax.scatter(self.glyc['s'],2*self.glyc['sp']/self.glyc['delta_t'],marker='+',c='red',label='patient')
            #
            ax.axhline(y=150, xmin=0,xmax=300,c='red',linewidth=0.5,linestyle='-')
            ax.axhline(y=55, xmin=0,xmax=300,c='red',linewidth=0.5,linestyle='-')
            ax.set_xlim([50,300])
            ax.set_ylim([0,300])
            ax.set_xlabel("initial plasma level (s')")
            ax.set_ylabel("s'|s,a")
            ax.grid('on')
            ax.legend(fontsize=10)
            ax.set_title('a. simulated results')
        
            #
            a = len(plasma[(plasma>55) & (plasma<150)])
            b = upper-a
            print(float(a)/upper)
            x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b), 100)
            dens = beta.pdf(x, a, b)
            ax1.plot(x, beta.pdf(x, a+1, b+1),'r-', lw=2, alpha=0.6, label='plasma control glucose pdf')
            ax1.set_xlim([0,1])
            ax1.grid('on')
            ax1.set_xlabel('p')
            ax1.set_title('b. pdf for control.')
            fig.show()
            fig.savefig('fig1.pdf')
            return plasma,dens
                
#####run 
# d=diary('data-26') patient reporting meals.
#fname data-35 is a winner
fname = 'data-35'  # choose a patient file
d=diary(fname)

d.glyc.fillna(0,inplace=True)
#

d.glyc.to_csv(fname+'_diary.csv') # save the file
#
data=d.glyc
df = data[['s','sp','reg','nph','lente','delta_t','lk','a']]
#axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
data=d.glyc
#
model = ols("sp~s+delta_t+reg+nph+lente",data).fit()  # fit model
print(model.summary().as_latex()) # observe coefficients
a,r,t = d.sim_solution(model)     # simulate
s,dens = d.trace_solution(t,model) # test
#print(model.summary())
