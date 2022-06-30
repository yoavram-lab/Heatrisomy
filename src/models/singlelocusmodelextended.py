import numpy as np
import json
import datetime
import signal,sys
import matplotlib.pyplot as plt

class SingleLocusModelExt:
    # k*μ is mutation rate of 2n genotype 
    def __init__(self, k):
        self.k=k
        
    # w1 is 2n+1 fitness
    # w2 is 2n+1* fitness
    # w3 is 2n* fitness
    
    #This method simulates follows the 4 states and records their frequency for every generation
    def run_simulations(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, fixation=0.95, max_gen=2500, seed=3, clonal_intf=True):
        curr_rand_state = np.random.get_state()
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 + (from tri+)
        #4 + (from wt)
        Mt = np.diag(np.repeat(0.,5))
        Mt[0][1] = aneuploidy_rate
        Mt[1][2] = mutation_rate
        Mt[2][3] = aneuploidy_loss_rate 
        Mt[0][4] = self.k*mutation_rate  
        w = [1., w1, w2, w3, w3]
        times, p, non_aneu_fix= self._simulation(N, w, Mt, repetitions=repetitions, max_gen=max_gen,clonal_intf=clonal_intf ,fixation=fixation)
        
        np.random.set_state(curr_rand_state)
        return (times, p, non_aneu_fix)   

    
    
    def _simulation(self, N, w, Mt ,repetitions=1000, fixation=0.95, max_gen=2500, clonal_intf=True):
        assert N > 0
        N = np.uint64(N)
        L = len(w)
        S = np.diag(w)
        
        if clonal_intf==False:
            Mnotr=Mt+1e-5
            Mnotr=Mnotr-1e-5
            Mnotr[0][1]=0
            Mnotr[0][4]=0
            Mnotr=Mnotr.transpose()
            np.fill_diagonal(Mnotr,0) #neutralling side effect
            np.fill_diagonal(Mnotr,1-Mnotr.sum(axis=0)) #side effect
            Enotr= Mnotr @ S

        Mt = Mt.transpose() 
        np.fill_diagonal(Mt,0) #neutralling side effect
        np.fill_diagonal(Mt,1-Mt.sum(axis=0)) #side effect

        Et = Mt @ S
  
        # rows are genotypes, cols are repretitions
        n = np.zeros((L, repetitions))
        n[0,:] = N    
        # which columns to update
        update = np.array([True] * repetitions)
        directmut = np.array([True] * repetitions)
        combfix=np.array([True] * repetitions)

        # follow time
        T = np.zeros(repetitions)
        t = 0
        # follow population mean fitness
        P = []
        E=Et

        while update.any():
            if t>=max_gen-1:
                break

            t += 1
            T[update] = t        
            p = n/N  # counts to frequencies
            P.append(p)
            p = E @ p  # natural selection + mutation   
            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            directmut[update] = (n[-1,update] < N*fixation)  # fixation of fittest genotype directly through mutation
            if clonal_intf==False:
                if (n[1,update]+n[-1,update])>=1:
                    E=Enotr
            update = (n[-2,:] < N*fixation)  # fixation of fittest genotype with aneuploidy
            update = update*directmut     # get all fixed repetitions into update
            combfix[update] = (n[-1,update]+n[-2,update] < N*fixation)  # fixation of fittest genotype as a combination
            update = update*combfix    # get all fixed repetitions into update
            
        #last generation update
        p = n/N 
        P.append(p)
        nona=(~directmut).nonzero()[0] #an array that stores the indices of the repetition where the 2n* is fixed through dircet mutations
        combfix=(~combfix).nonzero()[0] #an array that stores the indices of the repetition where the 2n* is fixed as a combination

        return T, P, (nona, combfix)
        #return T, P, nona
    
    # simulations only with waiting times and the frequency of 2n* through both trajectories (if fix_freq=True) as return 
    def run_simulations_time(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, max_gen=2500, fixation=0.95, seed=5,clonal_intf=True, fix_frequ=False):
        curr_rand_state = np.random.get_state()
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 + (from tri+)
        #4 + (from wt)
        M = np.diag(np.repeat(0.,5))
        M[0][1] = aneuploidy_rate
        M[1][2] = mutation_rate
        M[2][3] = aneuploidy_loss_rate 
        M[0][4] = self.k*mutation_rate  
        w = [1., w1, w2, w3, w3]
        times, non_aneu_fix, P= self._simulation_time(N, w, M, repetitions=repetitions, fixation=fixation, max_gen=max_gen,clonal_intf=clonal_intf, fix_frequ=fix_frequ)
        
        np.random.set_state(curr_rand_state)
        if fix_frequ is True:
            return (times, non_aneu_fix, P)
        else:
            return (times, non_aneu_fix)

    
    #have side effect on M, but not harmfull, can run twice simulation
    def _simulation_time(self, N, w, M ,repetitions=1000, fixation=0.95, max_gen=2500,clonal_intf=True, fix_frequ=False):
        assert N > 0
        N = np.uint64(N)
        L = len(w)
        S = np.diag(w)
                                
        if clonal_intf==False:
            Mnotr=M+1e-5
            Mnotr=Mnotr-1e-5
            Mnotr[0][1]=0
            Mnotr[0][4]=0
            Mnotr=Mnotr.transpose()
            np.fill_diagonal(Mnotr,0) #neutralling side effect
            np.fill_diagonal(Mnotr,1-Mnotr.sum(axis=0)) #side effect
            Enotr= Mnotr @ S                         

        M = M.transpose() 
        np.fill_diagonal(M,0) #neutralling side effect
        np.fill_diagonal(M,1-M.sum(axis=0)) #side effect

        E = M @ S

        # rows are genotypes, cols are repretitions
        n = np.zeros((L, repetitions))
        n[0,:] = N    
        # which columns to update
        update = np.array([True] * repetitions)
        directmut = np.array([True] * repetitions)
        combfix=np.array([True] * repetitions)

        # follow time
        T = np.zeros(repetitions)
        fixT=np.zeros(repetitions)
        t = 0

        P = []

        while update.any():
            if t>=max_gen-1:
                break
            
            t += 1
            T[update] = t   
            p = n/N  # counts to frequencies
            if clonal_intf==False:
                p[:,~clonal_int] = E @ p[:,~clonal_int]  # natural selection + mutation 
                p[:,clonal_int] = Enotr @ p[:,clonal_int]     
            else:
                p = E @ p

            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            directmut[update] = (n[-1,update] < N*fixation)  # fixation of fittest genotype directly through mutation
            if clonal_intf==False:
                clonal_int[update]=(n[1,update]+n[-1,update])>=1
            update = (n[-2,:] < N*fixation)  #fixation of fittest genotype with aneuploidy
            update = update*directmut  #get all fixed repetitions into update
            #solution to problem? 20.4.22
            combfix[update] = (n[-1,update]+n[-2,update] < N*fixation)  # fixation of fittest genotype as a combination
            update = update*combfix     #get all fixed repetitions into update
            
        #last generation update
        p = n/N 
        if fix_frequ is True: #Record the last frequencies of the two bins of 2n* if wanted
            P.append(p[3:5])
        nona=(~directmut).nonzero()[0]
        combfix=(~combfix).nonzero()[0]

        return T, (nona,combfix), P
        #return T, nona
    
        
    
    #Plot the frequency trajectories of the simulations
    def plot_progress(self, nparr, replicaid, state_names, colors, fixation = 0.95, ax=None, alpha=1, legend=True, xlim=True, ylim=(0,1), lw=None, g_o=False, logscale=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(14,5))

        #nparr[:,stateid][:,replicaid]
        #states_num = len(nparr[0])

        #finding last fixation index
        try:
            last = next(i for i,p in enumerate(nparr[:,-1][:,replicaid]) if p>=fixation)
        except StopIteration:
            last = nparr.shape[0]-1
        
        if legend is True:
            ind = -1
            for n,c in zip(state_names, colors):    
                ind+=1
                progress = nparr[:,ind][:last,replicaid]

                if g_o is False:
                    ax.plot(range(last), progress, label=n, color=c, alpha=alpha, lw=lw) 
                elif ind==3:
                    ax.plot(range(last), progress, label=n, color=c, alpha=.4, lw=lw, zorder=4)
                elif ind==2:
                    ax.plot(range(last), progress, label=n, color=c, alpha=alpha, lw=lw, zorder=5)
                else:
                    ax.plot(range(last), progress, label=n, color=c, alpha=alpha, lw=lw)
                
                    
        else:
            ind = -1
            for n,c in zip(state_names, colors):     
                ind+=1
                progress = nparr[:,ind][:last,replicaid]

                if g_o is False:
                    ax.plot(range(last), progress,  color=c, alpha=alpha, lw=lw) 
                elif ind==3:
                    ax.plot(range(last), progress,  color=c, alpha=.4, lw=lw, zorder=4)
                elif ind==2:
                    ax.plot(range(last), progress,  color=c, alpha=alpha, lw=lw, zorder=5)
                else:
                    ax.plot(range(last), progress,  color=c, alpha=alpha, lw=lw)
                
                    
                
        if xlim is not True:
            ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if legend is True:
            lg=ax.legend(loc='best')
            for lh in lg.legendHandles: 
                lh.set_alpha(1)
                lh.set_linewidth(1.5)
                
        if logscale is True:
            ax.set_yscale('log')
                
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        return ax
       # plt.show()
        
      