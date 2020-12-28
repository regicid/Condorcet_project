import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import multiprocessing as mtp
from tqdm import tqdm_notebook as tqdm

class CurtyMarsili(object):
    def __init__(self,z=0,z2 = 0,z3=0,a = 1, N=5000, p=0.52, m=11,γ = 0.05,γ2 = .05,σ = 10**-8,α_dandy = 1,n = 100,ω = 1,c = .01,selection_force=2,raoult=True,tqdm=False,T = 100000):
        #z is the initial proportion of followers (idem z2 -> proportion of dandys, z3 -> proportion of anti-conformists)
        self.N = N # Number of players
        self.ω = ω #Inventive to gather in-degrees
        self.p = p #Informed accuracy
        self.T = T #Number of iterations in the imitation phase
        self.tqdm = tqdm # Boolean, wether to use tqdm
        self.γ2 = γ2 #Accuracy update speed
        self.raoult = raoult #Wether to allow anti-conformists strategies
        self.m = m #Number of agents a follower listens to
        self.a = a #Incompressible probability to be chosen 
        self.c = c #Information searching fitness cost
        self.n = n #Computationnal stuff to speed things
        self.selection_force = selection_force #Number of birth/deaths at each time step
        self.σ = σ #Mutation rate
        self.γ = γ # Network update speed
        self.α_dandy = α_dandy # Originality's weighting for dandies
        self.N_f = int(np.round(N*z)) #Array recording what agents are follower
        self.N_α = int(np.round(N*z2)) #Idem for dandies
        self.anti_conformist = np.zeros(self.N,dtype="bool") #Idem for anti-conformists
        self.follower = np.zeros(self.N,dtype="bool")
        self.follower[:self.N_f] = True
        self.α = np.zeros(self.N,dtype='bool')
        self.α[:self.N_α] = True
        b = np.random.random(size=self.N)
        self.anti_conformist[b<z3] = True
        self.network = np.empty(shape = (self.N,m),dtype = "int16")
        self.network_scores = np.zeros(shape = (self.N,m))
        for i in range(self.N):
            self.network[i,] = np.random.choice(np.delete(range(N),i),size=m,replace=False)
        self.D = np.empty(N)
        self.D[self.follower] = np.random.choice([-1,1],p = [0.5, 0.5],size = self.follower.sum())
        self.D[~self.follower] = np.random.choice([-1,1],p = [1-p,p],size = N - self.follower.sum())
        self.prop_i = [] #Records the informed audience share
        in_deg = np.zeros(self.N,dtype="int")
        a = np.unique(self.network,return_counts=True)
        in_deg[a[0]] = a[1]
        self.N_f = [self.N_f] # Records the number of followers throughout time
        self.α_history = []
        self.f_history = []
        self.anti_history = []
        self.fitness_history = np.zeros(shape=(0,4))
        self.accuracy = .5*np.ones(shape=(self.N))
        self.q_history = []
    def compute_q(self):
        return np.mean(self.D[self.follower] > 0)
    def compute_pi(self):
        return np.mean(self.D > 0)
    def compute_informed(self):
        return np.mean(self.D[~self.follower] > 0)
    def iterate(self): # Iterative imitation process
        self.D[self.follower] = np.random.choice([-1,1],p = [0.5, 0.5],size = self.follower.sum())
        self.D[~self.follower] = np.random.choice([-1,1],p = [1-self.p,self.p],size = self.N - self.follower.sum())
        if self.follower.sum()>0:
            for t in range(self.T//self.n):
                #pick a random follower
                random_follower = np.random.choice(np.where(self.follower)[0],size = self.n)
                #get the choices of the group
                group_choices = self.D[self.network[random_follower,]]
                #align your choice with that of the majority
                avg_group_choice = np.mean(group_choices,axis=1)
                self.D[random_follower] = np.sign(avg_group_choice)*(1-2*self.anti_conformist[random_follower])
    def dynamics(self,T):
        self.fitness_history = np.vstack((self.fitness_history,np.zeros(shape=(T,4))))
        if self.tqdm:
            iter = tqdm(range(T))
        else:
            iter = range(T)
        for t in iter:
            # Now we update the networks (record scores, and get rid of the worst network member)
            in_deg = np.zeros(self.N,dtype="int")
            a = np.unique(self.network,return_counts=True)
            in_deg[a[0]] = a[1]
            #self.in_d = np.vstack((self.in_d,in_deg))
            self.network_scores += (self.D[self.network]>0) + self.α_dandy*np.broadcast_to(self.α,shape=(self.m,self.N)).transpose()*(self.D[self.network]
                                    - np.mean(self.D[self.network]))**2 - self.γ*self.network_scores
            a = np.where(np.random.random(size=self.N)< self.γ)[0]
            #Shuffle the network to avoid argmin issues
            z = np.random.permutation(self.m) 
            self.network = self.network[:,z]
            self.network_scores = self.network_scores[:,z]
            #select the poorest forecaster
            weakest_link = np.argmin(self.network_scores[a,],axis=1)
            #weakest_link = np.argmin(self.accuracy[self.network[a,]],axis=1)
            p = in_deg + self.a
            for i in range(len(a)):
                I = a[i]
                not_listened = np.where(~np.isin(np.arange(self.N),np.append(self.network[I,],I)))[0]
                p2 = p[not_listened]
                p2 = p2/p2.sum()
                new = np.random.choice(not_listened,p = p2)
                self.network[I,weakest_link[i]] = new
                self.network_scores[I,weakest_link[i]] = self.network_scores[I,].mean()
            b = np.random.random(size=self.N)
            self.α[b<self.σ] = 1 - self.α[b<self.σ]
            b = np.random.random(size=self.N)
            self.follower[b<self.σ] = 1 - self.follower[b<self.σ]
            if self.raoult:
                b = np.random.random(size=self.N)
                self.anti_conformist[b<self.σ] = 1 - self.anti_conformist[b<self.σ]
            self.accuracy += self.γ2*(self.D>0) -self.γ2*self.accuracy
            self.fitness = self.accuracy + self.ω*in_deg/self.N - self.c*(~self.follower)
            self.fitness /= self.fitness.sum()
            for j in range(self.selection_force):
                self.selection()
            self.N_f.append(self.follower.sum())
            self.α_history.append(self.α.mean())
            self.f_history.append(self.follower.mean())
            self.anti_history.append((self.anti_conformist*self.follower).mean())
            self.fitness_history[t,] = [self.fitness[~self.follower*~self.anti_conformist].mean(),self.fitness[~self.α*self.follower*~self.anti_conformist].mean(),self.fitness[self.α*self.follower*~self.anti_conformist].mean(),self.fitness[self.follower*self.anti_conformist].mean()]
            self.iterate()
            self.q_history.append(self.compute_q())
            self.prop_i.append(1-np.mean(self.follower[self.network]))
    def selection(self):
        t = len(self.q_history)
        i = np.random.choice(range(self.N))
        j = np.random.choice(range(self.N),p = self.fitness)
        self.α[i] = self.α[j]
        self.network[i,] = self.network[j,]
        self.network_scores[i,] = self.network_scores[j,]
        self.follower[i] = self.follower[j]
        self.anti_conformist[i] = self.anti_conformist[j]
        self.accuracy[i] = self.accuracy[j]
    def compressor(self):
        self.q = np.mean(self.q_history[-10**6:])
        self.f_history = pd.DataFrame(self.f_history).rolling(100).mean().values[::100,0]
        self.α_history = pd.DataFrame(self.α_history).rolling(100).mean().values[::100,0]
        self.prop_i = pd.DataFrame(self.prop_i).rolling(100).mean().values[::100,0]
        self.q_history = pd.DataFrame(self.q_history).rolling(100).mean().values[::100,0]
        self.anti_history = pd.DataFrame(self.anti_history).rolling(100).mean().values[::100,0]
        self.N_f= self.N_f[::100]
        self.fitness_history = pd.DataFrame(self.fitness_history).rolling(100).mean().values[::100,:]
