#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
num=8
bin_factor=10
identity="bin"
init_size=10000

def get_spec(filename):
	with open(filename) as f:
		i=f.read().split("\n")[:init_size]
		i=[np.array([float(z) for z in q.split(" ") if len(z)>0 ]) for q in i if len(q)>0]
		d=np.vstack(i).T
		return d



def timefiles(filename):
	dat=[map(float,r.strip().split(" ")) for r in open(filename).read().strip().split("\n")]
	datdat=[]
	time=[]
	for dd in dat:
		if len(dd)==1:
			datdat.append([])
			time.append(dd[0])
		else:
			datdat[-1].append(dd)
	datdat=np.vstack(datdat)
	datdat=datdat.reshape(len(time),datdat.shape[0]/len(time),datdat.shape[1])
	return time,datdat



def plot_spec(arr,index,norm_ind=1,force_over=False):
	def _plot_spec(ax,x,y,col="k"):
		ax.loglog()
		ax.plot(x,savgol_filter(y,13,2),color=col)


	cols=["k","r","g","b","c"]
	if type(arr)==type([]):
		if type(index)==type([]):
			if not force_over:
				f,axs=plt.subplots(len(index), sharex=True,sharey=True)

			for n,a in enumerate(arr):
				for w,ind in enumerate(index):
					if force_over:
						_plot_spec(plt,a[0],a[ind]/a[norm_ind],cols[n])
					else:
						_plot_spec(axs[w],a[0],a[ind]/a[norm_ind],cols[n])
		else:
			for n,a in enumerate(arr):
				_plot_spec(plt,a[0],a[index]/a[norm_ind],cols[n])
	else:
		if type(index)==type([]):
			if not force_over:
				f,axs=plt.subplots(len(index), sharex=True,sharey=True)

			for w,ind in enumerate(index):
				if force_over:
						_plot_spec(plt,arr[0],arr[ind]/arr[norm_ind])
				else:
					_plot_spec(axs[w],arr[0],arr[ind]/arr[norm_ind])
		else:
			_plot_spec(plt,arr[0],arr[index]/arr[norm_ind])


def main(num=8):
	spec_ex=False
	for fn in ["spec_extract_%i.out" % n for n in xrange(num)]:
		print fn
		if spec_ex is False:
			spec_ex=get_spec(fn)
		else:
			spec_ex+=get_spec(fn)
	spec_ex/=num

	spec_sc=False
	for fn in ["spec_scat_extract_%i.out" % n for n in xrange(num)]:
		print fn
		if spec_sc is False:
			spec_sc=get_spec(fn)
		else:
			spec_sc+=get_spec(fn)
	spec_sc/=num

	bin_spec_ex=np.zeros((spec_ex.shape[0],spec_ex.shape[1]/bin_factor))
	bin_spec_sc=np.zeros((spec_sc.shape[0],spec_sc.shape[1]/bin_factor))

	for i in  xrange(spec_ex.shape[1]/bin_factor):
		bin_spec_ex[:,i]=spec_ex[:,i*bin_factor:(i+1)*bin_factor].mean(axis=1)

	for i in  xrange(spec_sc.shape[1]/bin_factor):
		bin_spec_sc[:,i]=spec_sc[:,i*bin_factor:(i+1)*bin_factor].mean(axis=1)

	np.save("spec_"+identity+"_extract.out",bin_spec_ex)
	np.save("spec_"+identity+"_scat.out",bin_spec_sc)

	bin_spec_direct=bin_spec_ex.copy()
	bin_spec_direct[3:]-=bin_spec_sc[3:]
	np.save("spec_"+identity+"_direct.out",bin_spec_direct)

	np.savetxt("spec_"+identity+"_extract.out",bin_spec_ex.T,delimiter=" ")
	np.savetxt("spec_"+identity+"_scat.out",bin_spec_sc.T,delimiter=" ")
	np.savetxt("spec_"+identity+"_direct.out",bin_spec_direct.T,delimiter=" ")

	return bin_spec_ex,bin_spec_sc,bin_spec_direct
