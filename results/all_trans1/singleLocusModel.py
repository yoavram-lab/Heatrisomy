import numpy as np
import json
import datetime
import signal,sys
import matplotlib.pyplot as plt

class SingleLocusModel:
    # k*μ is mutation rate of 2n+1 genotype 
    def __init__(self, k, seed=20, stopon=2500, fixation=0.95):
        self.k=k
        self.seed = seed
        self.stopon = stopon
        self.fixation = fixation
        
    # w1 is 2n+1 fitness
    # w2 is 2n+1* fitness
    # w3 is 2n* fitness
    def run_simulations(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, seed=None):
        curr_rand_state = np.random.get_state()
        if seed is None:
            seed = self.seed
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 +
        M = np.diag(np.repeat(0.,4))
        M[0][1] = aneuploidy_rate
        M[1][0] = aneuploidy_rate
        M[1][2] = self.k*mutation_rate
        M[2][1] = self.k*mutation_rate
        M[2][3] = aneuploidy_loss_rate 
        M[3][2] = aneuploidy_loss_rate 
        M[0][3] = mutation_rate  
        M[3][0] = mutation_rate  
        w = [1., w1, w2, w3]
        times, p = self._simulation(N, w, M, repetitions=repetitions)
        
        np.random.set_state(curr_rand_state)
        return (times,p)   

    def grade_function2(self, times_p, a0=100, a=450,b=1700,c=2350):
        fixation = 0.95
        times , p = times_p
        nparr = np.array(p)
        s = nparr.shape[2] #total num of replicas
        if nparr.shape[0]<a:
            return 0 
        Tidx0 = (nparr[a0-1][1]+nparr[a0-1][2])<0.95  
        Tidx = (nparr[a-1][1]+nparr[a-1][2])>0.95  #fixated 2n+1 and 2n+1*
        Tidx = Tidx0 &Tidx
        Xidx = np.full(s,True) if nparr.shape[0]<b else (nparr[b-1,3,:]>0.95) 
        Yidx = np.full(s,True) if nparr.shape[0]<c else (nparr[c-1,3,:]>0.95)
        Yidx = ~Xidx&Yidx
        Tnum = Tidx.sum()
        if Tnum==0:
            return 0

        T = (Tnum/s)
        notXgivenT = (~Xidx&Tidx).sum()/Tnum
        notYgivenT = (~Yidx&Tidx).sum()/Tnum
        notXandnotYgivenT = (~Xidx&~Yidx&Tidx).sum()/Tnum 

        return -(T**4)*(1-(notXgivenT**4+notYgivenT**4-notXandnotYgivenT**4))


    def grade_function(self, times_p, a=450,b=1700,c=2350):
        fixation = self.fixation
        times , p = times_p
        nparr = np.array(p)
        s = nparr.shape[2] #total num of replicas
        if nparr.shape[0]<a:
            return 0 
        Tidx = (nparr[a-1][1]+nparr[a-1][2])>0.95  #fixated 2n+1 and 2n+1*
        Xidx = np.full(s,True) if nparr.shape[0]<b else (nparr[b-1,3,:]>0.95) 
        Yidx = np.full(s,True) if nparr.shape[0]<c else (nparr[c-1,3,:]>0.95)
        Yidx = ~Xidx&Yidx
        Tnum = Tidx.sum()
        if Tnum==0:
            return 0

        T = (Tnum/s)
        notXgivenT = (~Xidx&Tidx).sum()/Tnum
        notYgivenT = (~Yidx&Tidx).sum()/Tnum
        notXandnotYgivenT = (~Xidx&~Yidx&Tidx).sum()/Tnum 

        return -(T**4)*(1-(notXgivenT**4+notYgivenT**4-notXandnotYgivenT**4))

    def grade_function_ph(self, times_p, a=150, b=600):
        fixation = self.fixation
        times , p = times_p
        nparr = np.array(p)
        s = nparr.shape[2] #total num of replicas
        if nparr.shape[0]<a:
            return 0 
        Tidx = (nparr[a-1][1]+nparr[a-1][2])>0.95  #fixated 2n+1 and 2n+1*
        Xidx = np.full(s,True) if nparr.shape[0]<b else (nparr[b-1,3,:]>0.95) 
        Yidx = ~Xidx
        Tnum = Tidx.sum()
        if Tnum==0:
            return 0

        T = (Tnum/s)
        XgivenT = (Xidx&Tidx).sum()/Tnum
        YgivenT = (Yidx&Tidx).sum()/Tnum
        notXandnotYgivenT = (~Xidx&~Yidx&Tidx).sum()/Tnum 

        return -(T**4)*6*(XgivenT**2)*(YgivenT**2)

    def grade_function_no_aneuploidy(self, times_p, b=1700,c=2350):
        fixation = self.fixation
        times , p = times_p
        nparr = np.array(p)
        s = nparr.shape[2] #total num of replicas
        Xidx = np.full(s,True) if nparr.shape[0]<b else (nparr[b-1,3,:]>0.95) 
        Yidx = np.full(s,True) if nparr.shape[0]<c else (nparr[c-1,3,:]>0.95)
        Yidx = ~Xidx&Yidx

        notXgivenT = (~Xidx).sum()/s
        notYgivenT = (~Yidx).sum()/s
        notXandnotYgivenT = (~Xidx&~Yidx).sum()/s 

        return -(1-(notXgivenT**4+notYgivenT**4-notXandnotYgivenT**4))
    
    #have side effect on M, but not harmfull, can run twice simulation
    def _simulation(self, N, w, M ,repetitions=1000):
        fixation = self.fixation
                
        N = np.uint64(N)
        L = len(w)
        S = np.diag(w)

        M = M.transpose() 
        np.fill_diagonal(M,0) #neutralling side effect
        np.fill_diagonal(M,1-M.sum(axis=0)) #side effect

        E = M @ S

        # rows are genotypes, cols are repretitions
        n = np.zeros((L, repetitions))
        n[0,:] = N    
        # which columns to update
        update = np.array([True] * repetitions)
        # follow time
        T = np.zeros(repetitions)
        t = 0
        # follow population mean fitness
        W = []
        P = []

        while update.any():
            if t>=self.stopon-1:
                break

            t += 1
            T[update] = t        
            p = n/N  # counts to frequencies
    #         W.append(w.reshape(1, L) @ p)  # mean fitness
            P.append(p)
            p = E @ p  # natural selection + mutation        
            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            update = (n[-1,:] < N*fixation)  # fixation of fittest genotype
        #last generation update
        p = n/N 
        P.append(p)

        return T, P

    def plot_progress(self, nparr, replicaid, state_names=['2n','2n+1','2n+1*','2n*'], colors=['black','yellow','red','green'], fixation = 0.95):
        fig, ax = plt.subplots()

        #nparr[:,stateid][:,replicaid]
        #states_num = len(nparr[0])

        #finding last fixation index
        try:
            last = next(i for i,p in enumerate(nparr[:,-1][:,replicaid]) if p>=fixation)
        except StopIteration:
            last = nparr.shape[0]-1

        ind = -1
        for n,c in zip(state_names, colors):    
            ind+=1
            progress = nparr[:,ind][:last,replicaid]

            plt.plot(range(last), progress, label=n, color=c)

        plt.legend()
        plt.show()