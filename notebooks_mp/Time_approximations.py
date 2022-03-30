import numpy as np
import json
import datetime
import signal,sys
import matplotlib.pyplot as plt
from scipy.integrate import quad 
from functools import partial




    
### Most basic functions and approximations:
    
    # w1 is 2n+1 fitness
    # w2 is 2n+1* fitness
    # w3 is 2n* fitness
    
#def __init__(self):
#    self.T_kimura = np.vectorize(self.T_kimura0) 

#various relative fitnesses: s0=w3-1, s1=w1-1, s2=(w2-w1)/w1, s3=(w3-w2)/w2
def sv(w1,w2,w3):
    return w3-1, w1-1, (w2-w1)/w1, (w3-w2)/w2


#with Haldane fix_prob
def waitT(N, rate, s):
    if np.ndim(rate) == 0:
        if rate ==0:
            return 0
        return 1/(1-(1-rate*np.minimum(1,2*s))**N)
    rate=np.array([rate]).flatten()
    zerovec=(rate==0)
    retvec=np.array([0]*len(rate),dtype='float')
    retvec[~zerovec]=1/(1-(1-rate[~zerovec]*np.minimum(1,2*s))**N)
    return retvec

def min_waitT(N,mut_rate, aneu_rate, w1, w3):
    aneu=(1-(1-aneu_rate*np.minimum(1,2*sv(w1,1,w3)[1]))**N)
    mut=(1-(1-mut_rate*np.minimum(1,2*sv(w1,1,w3)[0]))**N)
    comp_para=1-(1-aneu)*(1-mut)
    return 1/comp_para
               


def waitT_K(N, rate, s):
    if np.ndim(rate) == 0:
        if rate ==0:
            return 0
        return 1/(1-(1-rate*np.minimum(1,2*s))**N)
    rate=np.array([rate]).flatten()
    zerovec=(rate==0)
    retvec=np.array([0]*len(rate))
    retvec[~zerovec]=1/(1-(1-rate[~zerovec]*np.expm1(-2 * 1 * s)/np.expm1(-2 * N * s))**N)
    return retvec


##Waiting time approximations until the first event happens that goes to fixation:

#with Haldane fix_prob
def waitgeoapprox(N, rate, s):
    rate =np.maximum(rate, 1e-16)
    return 1/(rate*N*np.minimum(1,2*s))



#with Kimura fix_prob
def waitgeoapproxKim(N, rate, s):
    rate =np.maximum(rate,1e-16) #if 0 in np.array([rate]):
        #return 0
    return np.expm1(-2 * N * s)/(rate*N*np.expm1(-2 * 1 * s) )

#prob that aneuploidy change happens first for the wild type                       
def probaneufirst(N, mut_rate,aneu_rate,w1,w3):
    aneu_rate =np.maximum(aneu_rate,1e-16)
    aneu = 1/waitT(N,aneu_rate,sv(w1,1,w3)[1])
    mut = 1/waitT(N,mut_rate,sv(w1,1,w3)[0])
    return aneu/(aneu + mut)

#prob that aneuploidy change happens first for the wild type with Kimura fixation probability:                      
def probaneufirst_K(N, mut_rate,aneu_rate,w1,w3):
    if aneu_rate == 0:
        return 0
    aneu = 1/waitT_K(N,aneu_rate,sv(w1,1,w3)[1])
    mut = 1/waitT_K(N,mut_rate,sv(w1,1,w3)[0])
    return aneu/(aneu + mut)

#prob that aneuploidy change happens first for the wild type                       
#def probaneufirst(N, mut_rate,aneu_rate,w1,w3):
#    aneu_rate =np.maximum(aneu_rate,1e-16)
#    aneu = 1/waitgeoapprox(N,aneu_rate,sv(w1,1,w3)[1])
#    mut = 1/waitgeoapprox(N,mut_rate,sv(w1,1,w3)[0])
#    return aneu/(aneu + mut)

#prob that aneuploidy change happens first for the wild type with Kimura fixation probability:                      
#def probaneufirst_K(N, mut_rate,aneu_rate,w1,w3):
#    if aneu_rate == 0:
#        return 0
#    aneu = 1/waitgeoapproxKim(N,aneu_rate,sv(w1,1,w3)[1])
#    mut = 1/waitgeoapproxKim(N,mut_rate,sv(w1,1,w3)[0])
#    return aneu/(aneu + mut)


##Fixation time approximations:

#deterministic approximation to fixation assumed at frequency N-1/N
def T_haldane_ffix(N, s):
    return 2 * np.log(N) / np.log(1+s)

#general deterministic approximation to fixation assumed at frequency 95%
def T_haldane(N, s, fixation_cutoff=0.95, n0=1):
    if fixation_cutoff < 1:
        T_ret=np.log(fixation_cutoff*(N-n0)/((1-fixation_cutoff)*n0)) / np.log(1+s)
    else:
        T_ret=2 * np.log(N) / np.log(1+s)
    return  np.maximum(T_ret,0)

#numerical simulation of the time to fixation with recurrent mutations
def time_evolution(p0, s, rate, tmax, cf=.95):
    p = [p0]

    for t in range(1, tmax):
        rec_p = p[t-1] * (1 + s) / (1 + p[t-1] * s)+(1-p[t-1])*2*s*rate
        p.append( rec_p)
        if rec_p>=cf:
            break

    return t

#Fixation time after Kimura
def integral(f, N, s, a, b):
    f = partial(f, N, s)    
    return quad(f, a, b, limit=100)[0]
def I1(N, s, x):
    if x == 1:
        return 0
    return (1 - np.exp(-2*N*s*x) - np.exp(-2 * N * s * (1 - x)) + np.exp(-2 * N *s)) / (x*(1-x))
def I2(N, s, x):
    if x == 0:
        return 0
    return -np.expm1(2 * N * s * x) * np.expm1(-2 * N * s * x) / (x * (1 - x))

#fix time with integral until 1->better perhaps with integration up to .95
#@np.vectorize
#def T_kimura(N, s):
    #x = 1 / N
    #J1 = -1.0 / (s * np.expm1(-2 * N * s)) * integral(I1, N, s, x, 1)
    #u = np.expm1(-2 * N * s * x) / np.expm1(-2 * N * s)
    #J2 = -1.0 / (s * np.expm1(-2 * N *s)) * integral(I2, N, s, 0, x)
    #return J1 + ((1 - u) / u) * J2
    
@np.vectorize
def T_kimura(N, s, fixation_cutoff=.95, n0=1):
    x = n0 / N
    J1 = -1.0 / (s * np.expm1(-2 * N * s)) * integral(I1, N, s, x, fixation_cutoff)
    u = np.expm1(-2 * N * s * x) / np.expm1(-2 * N * s)
    J2 = -1.0 / (s * np.expm1(-2 * N *s)) * integral(I2, N, s, 0, x)
    return J1 + ((1 - u) / u) * J2

###Direct mutation trajectory:

##Two main ones:

def T_mut(N, mut_rate, aneu_rate, w1, w3, fixation=False, fc0=.95, n00=1):
    wait_T=min_waitT(N, mut_rate, aneu_rate, w1, w3)
    if fixation is False:
        return wait_T
    return wait_T + T_haldane(N, sv(1,1,w3)[0], fc0, n00)

def T_mut_K(N, mut_rate, w3, fixation=False, fc0=.95, n00=1):
    wait_T_kim=waitT_K(N, mut_rate, sv(1,1,w3)[0])
    if fixation is False:
        return wait_T_kim
    return wait_T_kim + T_kimura(N, sv(1,1,w3)[0], fc0, n00)

##Two mixed approximation

def T_mut_H_K(N, mut_rate, w3, fixation=False):
    wait_T=waitT(N, mut_rate, sv(1,1,w3)[0])
    if fixation is False:
        return wait_T
    return wait_T + T_kimura(N, sv(1,1,w3)[0])

def T_mut_K_H(N, mut_rate, w3, fixation=False):
    wait_T_kim=waitT_K(N, mut_rate, sv(1,1,w3)[0])
    if fixation is False:
        return wait_T_kim
    return wait_T_kim + T_haldane(N, sv(1,1,w3)[0])


###Aneuploid trajectory:

##Two main ones:

def T_aneu(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False, fc1=.95,fc2=.95,fc3=.95, n01=1, n02=1, n03=1):
    wait_T1=min_waitT(N, mut_rate, aneu_rate, w1, w3)
    wait_T2=waitT(N, mut_rate, sv(w1,w2,w3)[2])
    wait_T3=waitT(N, aneu_rate, sv(w1,w2,w3)[3])
    wait_T123=wait_T1+wait_T2+wait_T3
    if fixation is False:
        return wait_T123
    fix_T1=T_haldane(N, sv(w1,w2,w3)[1], fc1, n01)
    fix_T2=T_haldane(N, sv(w1,w2,w3)[2], fc2, n02)
    fix_T3=T_haldane(N, sv(w1,w2,w3)[3], fc3, n03)
    fix_T123=fix_T1+fix_T2+fix_T3
    return wait_T123+fix_T123

def T_aneu_K(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False,fc1=.95,fc2=.95,fc3=.95, n01=1, n02=1, n03=1):
    wait_T1_K=waitT_K(N, aneu_rate, sv(w1,w2,w3)[1])
    wait_T2_K=waitT_K(N, mut_rate, sv(w1,w2,w3)[2])
    wait_T3_K=waitT_K(N, aneu_rate, sv(w1,w2,w3)[3])
    wait_T123_K=wait_T1_K+wait_T2_K+wait_T3_K
    if fixation is False:
        return wait_T123_K
    fix_T1_K=T_kimura(N, sv(w1,w2,w3)[1],fc1,n01)
    fix_T2_K=T_kimura(N, sv(w1,w2,w3)[2],fc2,n02)
    fix_T3_K=T_kimura(N, sv(w1,w2,w3)[3],fc3,n03)
    fix_T123_K=fix_T1_K+fix_T2_K+fix_T3_K
    return wait_T123_K+fix_T123_K

##Two mixed ones:

def T_aneu_H_K(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False):
    wait_T1=waitT(N, aneu_rate, sv(w1,w2,w3)[1])
    wait_T2=waitT(N, mut_rate, sv(w1,w2,w3)[2])
    wait_T3=waitT(N, aneu_rate, sv(w1,w2,w3)[3])
    if fixation is False:
        return wait_T1+wait_T2+wait_T3
    fix_T1_K=T_kimura(N, sv(w1,w2,w3)[1])
    fix_T2_K=T_kimura(N, sv(w1,w2,w3)[2])
    fix_T3_K=T_kimura(N, sv(w1,w2,w3)[3])
    return wait_T1+wait_T2+wait_T3+fix_T1_K+fix_T2_K+fix_T3_K

def T_aneu_K_H(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False):
    wait_T1_K=waitT_K(N, aneu_rate, sv(w1,w2,w3)[1])
    wait_T2_K=waitT_K(N, mut_rate, sv(w1,w2,w3)[2])
    wait_T3_K=waitT_K(N, aneu_rate, sv(w1,w2,w3)[3])
    if fixation is False:
        return wait_T1_K+wait_T2_K+wait_T3_K
    fix_T1=T_haldane(N, sv(w1,w2,w3)[1])
    fix_T2=T_haldane(N, sv(w1,w2,w3)[2])
    fix_T3=T_haldane(N, sv(w1,w2,w3)[3])
    return wait_T1_K+wait_T2_K+wait_T3_K+fix_T1+fix_T2+fix_T3


###Total weighted time:

##Two main approximations:

def T(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False, fc0=.95, fc1=.95, fc2=.95, fc3=.95, n00=1, n01=1, n02=1, n03=1):
    return probaneufirst(N,mut_rate,aneu_rate,w1,w3)*T_aneu(N, mut_rate, aneu_rate, w1, w2, w3, fixation, fc1, fc2, fc3, n01, n02, n03)+(1-probaneufirst(N,mut_rate,aneu_rate,w1,w3))*T_mut(N, mut_rate, aneu_rate, w1, w3, fixation, fc0, n00)

def T_K(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False, fc0=.95, fc1=.95, fc2=.95, fc3=.95, n00=1, n01=1, n02=1, n03=1):
    return probaneufirst(N,mut_rate,aneu_rate,w1,w3)*T_aneu_K(N, mut_rate, aneu_rate, w1, w2, w3, fixation, fc1, fc2, fc3, n01, n02, n03)+(1-probaneufirst(N,mut_rate,aneu_rate,w1,w3))*T_mut_K(N, mut_rate, w3, fixation, fc0, n00)

##Two mixed approximations:

def T_H_K(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False):
    return probaneufirst(N,mut_rate,aneu_rate,w1,w3)*T_aneu_H_K(N, mut_rate, aneu_rate, w1, w2, w3, fixation)+(1-probaneufirst(N,mut_rate,aneu_rate,w1,w3))*T_mut_H_K(N, mut_rate, w3, fixation)

def T_K_H(N, mut_rate, aneu_rate, w1, w2, w3, fixation=False):
    return probaneufirst(N,mut_rate,aneu_rate,w1,w3)*T_aneu_K_H(N, mut_rate, aneu_rate, w1, w2, w3, fixation)+(1-probaneufirst(N,mut_rate,aneu_rate,w1,w3))*T_mut_K_H(N, mut_rate, w3, fixation)

    
##First clonal interference approximations:    
def extra_conv(N,rate,s):
    return N*np.log(N)*rate/s

def interf_dirmut(N, mut_rate, aneu_rate, w1, w3):
    return extra_conv(N,mut_rate,sv(w1,1,w3)[1])*(1-probaneufirst(N,mut_rate, aneu_rate, w1, w3))

def interf_dirmut2(N, mut_rate, aneu_rate, w1, w3):
    return extra_conv(N,mut_rate,sv(w1,1,w3)[1])*(1-2*sv(w1,1,w3)[0]*(1-(1-mut_rate)**N)/(2*sv(w1,1,w3)[0]*(1-(1-mut_rate)**N)+2*sv(w1,1,w3)[1]*(1-(1-aneu_rate)**N)))

def prob_no_clo(N, mut_rate, aneu_rate, w1, w3):
    return np.exp(-interf_dirmut(N, mut_rate, aneu_rate, w1, w3))
    
def prob_no_clo2(N, mut_rate, aneu_rate, w1, w3):
    return np.exp(-interf_dirmut2(N, mut_rate, aneu_rate, w1, w3))
    
    
    
    