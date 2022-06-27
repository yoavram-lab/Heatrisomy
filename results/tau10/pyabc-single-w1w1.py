import pyabc
import scipy
import tempfile
import os
os.environ["OMP_NUM_THREADS"] = "1" #for numpy
import numpy as np
from model1.singleLocusModel import SingleLocusModel
import sys
from datetime import timedelta, datetime
from shutil import copyfile, copytree, rmtree
from multiprocessing import  Value


N = 6.425*10**6 # 1.6M - 2004.8M in Yona experiment 8/(1.0/(np.array([2**i for i in range(8)])*1.6)).sum()
GLOBAL_SEED = 123
# K = 1.03125
# A,B,C = 50, 300, 2200

PRIOR_THRESHOLD = 0.01
MAX_POPULATIONS = 50 # stop criterion: num of abc iterations
A,B,C = 450, 1700, 2350
REPS1 = int(sys.argv[1])
SIM_SEED = int(sys.argv[2]) #was 20
K = float(sys.argv[3])
MIN_EPSILON = float(sys.argv[4]) #stop criterion: was 0.25
DIR_NAME = sys.argv[5]
CPUS = int(sys.argv[6])

model1 = SingleLocusModel(k=K)

class W_RV(pyabc.RVBase):
    def __init__(self):
        l = np.load('./data/evo39_fitness_39deg.npz')
        self.kde = scipy.stats.gaussian_kde(l['arr_0'])
    def rvs(self, *args, **kwargs):
        return self.kde.resample(1)[0][0]
    def pdf(self, x, *args, **kwargs):
        return self.kde.pdf(x)[0]
    def copy(self):
        raise NotImplementedError('copy')
    def pmf(self, x, *args, **kwargs):
        raise NotImplementedError('pmf')
    def cdf(self, x, *args, **kwargs):
        raise NotImplementedError('cdf')

passed_count = Value('i', 0)

class CustomDistribution1(pyabc.Distribution):
    def rvs(self):
        global passed_count
        print('rvs running')
        found = False
        while not found:
            ans = super().rvs()
            grade = 1-abc_model1(ans,reps=REPS1,toprint=False)['res']
            if grade>PRIOR_THRESHOLD:
                found = True

        with passed_count.get_lock():
            passed_count.value += 1
        print('passed1', passed_count.value, ans)
        return ans


def abc_model1(parameter, reps=REPS1, toprint=True):
    #trisomy gain and loss are the same, therefore parameter.p2_tr twice
    w1 = parameter.p3_w1
    w2 = parameter.p4_w2
    w3 = parameter.p5_w3
    x = [parameter.p1_mr, parameter.p2_tr, parameter.p2_tr, w1, w2, w3]
    # if toprint:
        # print(x)
    if not x[3]<x[4]<x[5] or x[0]<0 or x[1]<0 or x[2]<0 or x[3]<1 or x[4]<1 or x[5]<1:
        # x[3]<x[4] because otherwise trisomy will not be reached.  x[4]<x[5] by Fig4A (anyway it should be x[3]<x[5]).
        return {'res':1}

    times_p = model1.run_simulations(N, x[0], x[1], x[2], x[3], x[4], x[5], repetitions=REPS1, seed=SIM_SEED)
    grade = 1+model1.grade_function2(times_p, a0=200, a=A, b=B, c=C)
    print('model1', 1-grade, x)
    return {'res':grade}

def distance(x, y):
    return x['res']

prior1 = CustomDistribution1(p1_mr=pyabc.RV("uniform", 10.0**-9, 10.0**-5-10.0**-9)
                            ,p2_tr=pyabc.RV("uniform", 10.0**-6, 10.0**-2-10.0**-6)
                            ,p3_w1=W_RV()
                            ,p4_w2=W_RV()
                            ,p5_w3=W_RV())

sampler = pyabc.sampler.MulticoreParticleParallelSampler(n_procs=CPUS)
if (CPUS==1):
    sampler = pyabc.sampler.SingleCoreSampler()
adaptive_population = pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=200, mean_cv=0.25)
abc = pyabc.ABCSMC(abc_model1, prior1, distance, sampler = sampler, population_size=adaptive_population)


now = datetime.now().strftime('%Y-%m-%d')
dir_name = now+'-'+DIR_NAME
path = '~/sim-data/single-model-abc/'+dir_name
path = os.path.expanduser(path)
if not os.path.exists(path):
	os.makedirs(path)
copyfile(sys.argv[0], os.path.join(path, sys.argv[0])) 
copyfile('model1/singleLocusModel.py', os.path.join(path, 'singleLocusModel.py'))
np.savetxt(path+'/params',sys.argv,fmt='%s',newline=' ')


history = abc.new('sqlite:///'+path+'/'+dir_name+'.db')
np.random.seed(GLOBAL_SEED)
history = abc.run(minimum_epsilon=MIN_EPSILON, max_nr_populations=MAX_POPULATIONS)