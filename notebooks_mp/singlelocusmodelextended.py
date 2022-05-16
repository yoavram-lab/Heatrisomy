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
    def run_simulations(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, fixation=0.95, max_gen=2500, seed=3, clonal_intf=True):
        curr_rand_state = np.random.get_state()
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 +
        Mt = np.diag(np.repeat(0.,5))
        Mt[0][1] = aneuploidy_rate
        Mt[1][2] = mutation_rate
        Mt[2][3] = aneuploidy_loss_rate 
        Mt[0][4] = self.k*mutation_rate  
        w = [1., w1, w2, w3, w3]
        times, p, non_aneu_fix= self._simulation(N, w, Mt, repetitions=repetitions, max_gen=max_gen,clonal_intf=clonal_intf ,fixation=fixation)
        
        np.random.set_state(curr_rand_state)
        return (times, p, non_aneu_fix)   

    
    #have side effect on M, but not harmfull, can run twice simulation
    def _simulation(self, N, w, Mt ,repetitions=1000, fixation=0.95, max_gen=2500,clonal_intf=True):
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
        #clonal_int=np.array([False]*repetitions)
        combfix=np.array([True] * repetitions)
        fixbool=np.array([False]*repetitions)
        # follow time
        T = np.zeros(repetitions)
        t = 0
        # follow population mean fitness
        W = []
        P = []
        E=Et

        while update.any():
            if t>=max_gen-1:
                break

            t += 1
            T[update] = t        
            p = n/N  # counts to frequencies
    #         W.append(w.reshape(1, L) @ p)  # mean fitness
            P.append(p)
            p = E @ p  # natural selection + mutation   
            #p[-1,clonal_int]=0
            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            directmut[update] = (n[-1,update] < N*fixation)  # fixation of fittest genotype directly through mutation
            if clonal_intf==False:
                if (n[1,update]+n[-1,update])>=1:
                    E=Enotr
            #clonal_int[update]=(n[1,update]+n[-1,update])>=1
            update = (n[-2,:] < N*fixation)  # fixation of fittest genotype with aneuploidy
            update = update*directmut
            combfix[update] = (n[-1,update]+n[-2,update] < N*fixation)  # fixation of fittest genotype as a combination
            update = update*combfix
        #last generation update
        
        p = n/N 
        P.append(p)
        nona=(~directmut).nonzero()[0]
        combfix=(~combfix).nonzero()[0]

        return T, P, (nona, combfix)
        #return T, P, nona
    
    # simulations only with waiting time as return
    def run_simulations_time(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, max_gen=2500, fixation=0.95, seed=5,clonal_intf=True):
        curr_rand_state = np.random.get_state()
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 +
        M = np.diag(np.repeat(0.,5))
        M[0][1] = aneuploidy_rate
        M[1][2] = mutation_rate
        M[2][3] = aneuploidy_loss_rate 
        M[0][4] = self.k*mutation_rate  
        w = [1., w1, w2, w3, w3]
        times, non_aneu_fix= self._simulation_time(N, w, M, repetitions=repetitions, fixation=fixation, max_gen=max_gen,clonal_intf=clonal_intf)
        
        np.random.set_state(curr_rand_state)
        return (times, non_aneu_fix)

    
    #have side effect on M, but not harmfull, can run twice simulation
    def _simulation_time(self, N, w, M ,repetitions=1000, fixation=0.95, max_gen=2500,clonal_intf=True):
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
        clonal_int=np.array([False]*repetitions)
        fixbool=np.array([False]*repetitions)
        # follow time
        T = np.zeros(repetitions)
        fixT=np.zeros(repetitions)
        t = 0
        # follow population mean fitness
        W = []
        P = []

        while update.any():
            if t>=max_gen-1:
                break
            
            t += 1
            T[update] = t   
            p = n/N  # counts to frequencies
            #W.append(w.reshape(1, L) @ p)  # mean fitness
            #P.append(p)
            if clonal_intf==False:
                p[:,~clonal_int] = E @ p[:,~clonal_int]  # natural selection + mutation 
                p[:,clonal_int] = Enotr @ p[:,clonal_int]     
            else:
                p = E @ p
            #p[-1,clonal_int]=0
            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            directmut[update] = (n[-1,update] < N*fixation)  # fixation of fittest genotype directly through mutation
            if clonal_intf==False:
                clonal_int[update]=(n[1,update]+n[-1,update])>=1
            #clonal_int[update]=np.logical_and((n[1,update]+n[2,update]>N*clonal_block_perc),(n[-1,update]<N*clonal_block_perc))
            update = (n[-2,:] < N*fixation)  # fixation of fittest genotype with aneuploidy
            update = update*directmut
            #solution to problem? 20.4.22
            combfix[update] = (n[-1,update]+n[-2,update] < N*fixation)  # fixation of fittest genotype as a combination
            update = update*combfix
            
        #last generation update
        p = n/N 
        #P.append(p)
        nona=(~directmut).nonzero()[0]
        combfix=(~combfix).nonzero()[0]

        return T, (nona,combfix)
        #return T, nona
    
        
    # simulations only with waiting time as return
    def run_simulations_time_fixT(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, max_gen=2500, fixation=0.95, seed=5,clonal_intf=True):
        curr_rand_state = np.random.get_state()
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 +
        M = np.diag(np.repeat(0.,5))
        M[0][1] = aneuploidy_rate
        M[1][2] = mutation_rate
        M[2][3] = aneuploidy_loss_rate 
        M[0][4] = self.k*mutation_rate  
        w = [1., w1, w2, w3, w3]
        times, non_aneu_fix, fixT= self._simulation_time_fixT(N, w, M, repetitions=repetitions, fixation=fixation, max_gen=max_gen,clonal_intf=clonal_intf)
        
        np.random.set_state(curr_rand_state)
        return (times, non_aneu_fix, fixT)

    
    
    def _simulation_time_fixT(self, N, w, M ,repetitions=1000, fixation=0.95, max_gen=2500,clonal_intf=True):
        assert N > 0
        N = np.uint64(N)
        L = len(w)
        S = np.diag(w)
                                
        if clonal_intf==False:
            Mnotr=np.diag(np.repeat(0.,5))
            Mnotr[1][2]=M[1][2]
            Mnotr[2][3]=M[2][3]
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
        clonal_int=np.array([False]*repetitions)
        fixbool=np.array([False]*repetitions)
        # follow time
        T = np.zeros(repetitions)
        fixT=np.zeros(repetitions)
        t = 0
        # follow population mean fitness
        W = []
        P = []

        while update.any():
            if t>=max_gen-1:
                break
            
            t += 1
            T[update] = t   
            
            fixT[fixbool*update] +=1
            fixT[~fixbool] =0
            p = n/N  # counts to frequencies
            #W.append(w.reshape(1, L) @ p)  # mean fitness
            #P.append(p)
            if clonal_intf==False:
                p[:,~clonal_int] = E @ p[:,~clonal_int]  # natural selection + mutation 
                p[:,clonal_int] = Enotr @ p[:,clonal_int]     
            else:
                p = E @ p
            #p[-1,clonal_int]=0
            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            directmut[update] = (n[-1,update] < N*fixation)  # fixation of fittest genotype directly through mutation
            fixbool[update]=(n[1,update]+n[2,update]+n[3,update]+n[4,update])>=1
            if clonal_intf==False:
                clonal_int[update]=(n[1,update]+n[-1,update])>=1
            #clonal_int[update]=np.logical_and((n[1,update]+n[2,update]>N*clonal_block_perc),(n[-1,update]<N*clonal_block_perc))
            update = (n[-2,:] < N*fixation)  # fixation of fittest genotype with aneuploidy
            update = update*directmut
            #combfix[update] = (n[-1,update]+n[-2,update] < N*fixation)  # fixation of fittest genotype as a combination
            #update = update*combfix
        #last generation update
        p = n/N 
        #P.append(p)
        nona=(~directmut).nonzero()[0]
        #combfix=(~combfix).nonzero()[0]

        #return T, (nona,combfix)
        return T, nona, fixT
    
    
    # simulations only with waiting time as return
    def run_simulations_time_hyptest(self, N, mutation_rate, aneuploidy_rate, aneuploidy_loss_rate, w1, w2, w3, repetitions=1000, max_gen=2500, fixation=0.95, seed=5, clonal_aneu=1, clonal_mut=1):
        curr_rand_state = np.random.get_state()
        np.random.seed(seed)

        #0 wt
        #1 tri
        #2 tri+
        #3 +
        M = np.diag(np.repeat(0.,5))
        M[0][1] = aneuploidy_rate
        M[1][2] = mutation_rate
        M[2][3] = aneuploidy_loss_rate 
        M[0][4] = self.k*mutation_rate  
        w = [1., w1, w2, w3, w3]
        times, non_aneu_fix= self._simulation_time_hyptest(N, w, M, repetitions=repetitions, fixation=fixation, max_gen=max_gen, clonal_aneu=1, clonal_mut=1)
        
        np.random.set_state(curr_rand_state)
        return (times, non_aneu_fix)

    
    #have side effect on M, but not harmfull, can run twice simulation
    def _simulation_time_hyptest(self, N, w, M ,repetitions=1000, fixation=0.95, max_gen=2500,clonal_aneu=1,clonal_mut=1):
        assert N > 0
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
        directmut = np.array([True] * repetitions)
        combfix=np.array([True] * repetitions)
        clonal_int_aneu=np.array([False]*repetitions)
        clonal_int_mut=np.array([False]*repetitions)
        # follow time
        T = np.zeros(repetitions)
        t = 0
        # follow population mean fitness
        W = []
        P = []

        while update.any():
            if t>=max_gen-1:
                break

            t += 1
            T[update] = t        
            p = n/N  # counts to frequencies
    #         W.append(w.reshape(1, L) @ p)  # mean fitness
            #P.append(p)
            p = E @ p  # natural selection + mutation        
            p[-1,clonal_int_aneu]=0
            p[1,clonal_int_mut]=0
            p /= p.sum(axis=0)  # mean fitness
            for j in update.nonzero()[0]:
                # random genetic drift
                n[:,j] = np.random.multinomial(N, p[:,j])
            directmut[update] = (n[-1,update] < N*fixation)  # fixation of fittest genotype directly through mutation
            #clonal_int[update]=(n[1,update]+n[2,update]>N*clonal_block_perc & n[-1,update]<N*clonal_block_perc)
            clonal_int_aneu[update]=(n[1,update]+n[2,update]>=clonal_aneu)
            clonal_int_mut[update]=(n[-1,update]>=clonal_mut)
            update = (n[-2,:] < N*fixation)  # fixation of fittest genotype with aneuploidy
            update = update*directmut
            #combfix[update] = (n[-1,update]+n[-2,update] < N*fixation)  # fixation of fittest genotype as a combination
            #update = update*combfix
        #last generation update
        p = n/N 
        #P.append(p)
        nona=(~directmut).nonzero()[0]
        #combfix=(~combfix).nonzero()[0]

        #return T, (nona,combfix)
        return T, nona
    
    
    
    
    
    def meandiffT(self, fullarr, diffT):
        means0 = []
        means1 = []
        for j in range(len(fullarr)):
            update = np.array([True] * len(fullarr[j]))
            update[diffT[j]] = False
            if update.any(): 
                m0=np.mean(fullarr[j][update])
            else:
                m0=0    
            m1=np.mean(fullarr[j][~update])
            means0.append(m0)
            means1.append(m1)
        return (means0, means1)
    
    def meanstddiffT(self, fullarr, diffT):
        means0 = []
        means1 = []
        std0 = []
        std1 = []
        for j in range(len(fullarr)):
            update = np.array([True] * len(fullarr[j]))
            update[diffT[j]] = False
            if update.any(): 
                m0=np.mean(fullarr[j][update])
                sd0=np.std(fullarr[j][update])
            else:
                m0=0 
                sd0=0
            if (~update).any():
                m1=np.mean(fullarr[j][~update])
                sd1=np.std(fullarr[j][~update])
            else:
                m1=0
                sd1=0
            means0.append(m0)
            means1.append(m1)
            std0.append(sd0)
            std1.append(sd1)
        return (means0, means1), (std0,std1)

    def meanpercerrdiffT(self, fullarr, diffT):
        means_aneu = []
        means_mut = []
        err0595_aneu = []
        err0595_mut = []
        for j in range(len(fullarr)):
            update = np.array([True] * len(fullarr[j]))
            update[diffT[j]] = False
            if update.any(): 
                m0=np.mean(fullarr[j][update])
                e0=[m0-np.percentile(fullarr[j][update],5),np.percentile(fullarr[j][update],95)-m0]
            else:
                m0=0 
                e0=[0,0]
            if (~update).any():
                m1=np.mean(fullarr[j][~update])
                e1=[m1-np.percentile(fullarr[j][~update],5),np.percentile(fullarr[j][~update],95)-m1]
            else:
                m1=0
                e1=[0,0]
            means_aneu.append(m0)
            means_mut.append(m1)
            err0595_aneu.append(e0)
            err0595_mut.append(e1)
        return (means_aneu, means_mut), (np.array(err0595_aneu).transpose(),np.array(err0595_mut).transpose())

    
    def plot_progress(self, nparr, replicaid, state_names, colors, fixation = 0.95, ax=None, alpha=1, legend=True, xlim=True, ylim=(0,1), lw=None, g_o=False, logscale=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

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

            if g_o is False:
                ax.plot(range(last), progress, label=n, color=c, alpha=alpha, lw=lw) 
            elif ind==3:
                ax.plot(range(last), progress, label=n, color=c, alpha=.4, lw=lw, zorder=4)
            elif ind==2:
                ax.plot(range(last), progress, label=n, color=c, alpha=alpha, lw=lw, zorder=5)
            else:
                ax.plot(range(last), progress, label=n, color=c, alpha=alpha, lw=lw)
                
        if xlim is not True:
            ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if legend is True:
            lg=ax.legend(loc='best')
            for lh in lg.legendHandles: 
                lh.set_alpha(1)
                
        if logscale is True:
            ax.set_yscale('log')
                
        ax.set_xlabel('Time in generations')
        ax.set_ylabel('Frequency')
        return ax
       # plt.show()
    