import numpy as np
import pandas as pd
import glob
import matplotlib
from datetime import *
import csv
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import stats
import math
from scipy.optimize import curve_fit
from matplotlib import cm

def dg(cell):
	global tzero
	global dgtotal
	data = pd.read_csv("example.csv", index_col=None)
	coc3=data.loc[data["Well Position"]==cell]

	def func(x, A2, A1, x0, dx):
		return A2+(A1-A2)/(1+np.exp((x-x0)/dx))

	popt, pcov= curve_fit(func, coc3['Boltzmann Temperature'], coc3['Boltzmann Fluorescence'],p0=[80000, 80000, 70, .9])

	# print (cell)

	fmax=popt[0]
	fmin=popt[1]
	tm=popt[2]+273
	m=popt[3]

	coc3.drop(coc3.tail(10).index, inplace=True)
	coc3=coc3.iloc[10:]
	coc3["Boltzmann Temperature"]=coc3["Boltzmann Temperature"]+273
	dgtotal=[]
	index=np.array([1])
	for y in range(1, len(coc3)-1):
		pf=1-((coc3["Boltzmann Fluorescence"].iloc[index]-fmin)/(fmax-fmin))
		index=index+1
		pu=1-pf
		ku=float(pu/pf)
		# ku=float(pu/(1-pu))
		dg=-8.31*coc3["Boltzmann Temperature"].iloc[index]*np.log(ku)
		dg=float(dg)
		dgtotal.append(dg)
	global dgzero
	dgtotal=np.array(dgtotal)

	coc3=coc3.iloc[ :-2]

	slope, intercept, r_value, p_value, std_err = stats.linregress(coc3["Boltzmann Temperature"], dgtotal)
	
#define your tzero here
	tzero=343.73268

	dgzero=slope*tzero+intercept

	global linregressX
	global linregressY
	linregressX=range(340,349,1)
	linregressY=[]
	for value in linregressX:
		dg_intercept=slope*value+intercept
		linregressY.append(dg_intercept)
	
#if you want to see the boltzman fit, put this bit back in	
	# fig=plt.figure()
	# fig.suptitle(cell)

	# ax1=fig.add_subplot(111)	
	# ax1.scatter(coc3["Boltzmann Temperature"], dgtotal)
	# ax1.scatter(tzero, dgzero)
	# plt.show()
	# print (r_value**2)
	# plt.savefig("dg{cell}.png".format(cell=cell))
	global florgraph
	florgraph=coc3["Boltzmann Temperature"]

	return dgzero
	return florgraph
	return tzero
	return dgtotal
	return linregressX
	return linregressY


# enter here what concentration you want to use.
celllistA=[ "A05"]
celllistB=[ "B05"]
celllistC=[ "C05"]
celllistD=[ "D05"]
celllistE=[ "E05"]
celllistF=[ "F05"]

dgzerototal=[]
tzerograph=[]
florgrapha=[]
dgtotalgraph=[]
linregressXgraph=[]
linregressYgraph=[]

for value in celllistA:
	dgzero=dg(value)
	dgzerototal.append(dgzero)
	tzerograph.append(tzero)
	florgrapha.append(florgraph)
	dgtotalgraph.append(dgtotal)
	linregressXgraph.append(linregressX)
	linregressYgraph.append(linregressY)

florgrapha=np.array(florgrapha)
dgtotalgraph=np.array(dgtotalgraph)
tzerograph=np.array(tzerograph)
dgzerototal=np.array(dgzerototal)
linregressYgraph=np.array(linregressYgraph)
linregressXgraph=np.array(linregressXgraph)

dgzerototalb=[]
tzerographb=[]
florgraphab=[]
dgtotalgraphb=[]
linregressXgraphb=[]
linregressYgraphb=[]

for value in celllistB:
	dgzero=dg(value)
	dgzerototalb.append(dgzero)
	tzerographb.append(tzero)
	florgraphab.append(florgraph)
	dgtotalgraphb.append(dgtotal)
	linregressXgraphb.append(linregressX)
	linregressYgraphb.append(linregressY)

florgraphab=np.array(florgraphab)
dgtotalgraphb=np.array(dgtotalgraphb)
tzerographb=np.array(tzerographb)
dgzerototalb=np.array(dgzerototalb)
linregressYgraphb=np.array(linregressYgraphb)
linregressXgraphb=np.array(linregressXgraphb)

dgzerototalc=[]
tzerographc=[]
florgraphac=[]
dgtotalgraphc=[]
linregressXgraphc=[]
linregressYgraphc=[]

for value in celllistC:
	dgzero=dg(value)

	dgzerototalc.append(dgzero)
	tzerographc.append(tzero)
	florgraphac.append(florgraph)
	dgtotalgraphc.append(dgtotal)
	linregressXgraphc.append(linregressX)
	linregressYgraphc.append(linregressY)

florgraphac=np.array(florgraphac)
dgtotalgraphc=np.array(dgtotalgraphc)
tzerographc=np.array(tzerographc)
dgzerototalc=np.array(dgzerototalc)
linregressYgraphc=np.array(linregressYgraphc)
linregressXgraphc=np.array(linregressXgraphc)

dgzerototald=[]
tzerographd=[]
florgraphad=[]
dgtotalgraphd=[]
linregressXgraphd=[]
linregressYgraphd=[]
for value in celllistD:
	dgzero=dg(value)
	dgzerototald.append(dgzero)
	tzerographd.append(tzero)
	florgraphad.append(florgraph)
	dgtotalgraphd.append(dgtotal)
	linregressXgraphd.append(linregressX)
	linregressYgraphd.append(linregressY)

florgraphad=np.array(florgraphad)
dgtotalgraphd=np.array(dgtotalgraphd)
tzerographd=np.array(tzerographd)
dgzerototald=np.array(dgzerototald)
linregressYgraphd=np.array(linregressYgraphd)
linregressXgraphd=np.array(linregressXgraphd)

dgzerototale=[]
tzerographe=[]
florgraphae=[]
dgtotalgraphe=[]
linregressXgraphe=[]
linregressYgraphe=[]
for value in celllistE:
	dgzero=dg(value)
	dgzerototale.append(dgzero)
	tzerographe.append(tzero)
	florgraphae.append(florgraph)
	dgtotalgraphe.append(dgtotal)
	linregressXgraphe.append(linregressX)
	linregressYgraphe.append(linregressY)

florgraphae=np.array(florgraphae)
dgtotalgraphe=np.array(dgtotalgraphe)
tzerographe=np.array(tzerographe)
dgzerototale=np.array(dgzerototale)
linregressYgraphe=np.array(linregressYgraphe)
linregressXgraphe=np.array(linregressXgraphe)

dgzerototalf=[]

tzerographf=[]
florgraphaf=[]
dgtotalgraphf=[]
linregressXgraphf=[]
linregressYgraphf=[]

for value in celllistF:
	dgzero=dg(value)
	dgzerototalf.append(dgzero)
	tzerographf.append(tzero)
	florgraphaf.append(florgraph)
	dgtotalgraphf.append(dgtotal)
	linregressXgraphf.append(linregressX)
	linregressYgraphf.append(linregressY)

florgraphaf=np.array(florgraphaf)
dgtotalgraphf=np.array(dgtotalgraphf)
tzerographf=np.array(tzerographf)
dgzerototalf=np.array(dgzerototalf)
linregressYgraphf=np.array(linregressYgraphf)
linregressXgraphf=np.array(linregressXgraphf)
fig=plt.figure()
ax1=fig.add_subplot(111)


colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', '.5']

for i in range(len(celllistA)):
	line1 =ax1.scatter(florgrapha[i], dgtotalgraph[i]/1000, color=colors[i], label="Inline label", s=12)
	line2= ax1.scatter(florgraphab[i], dgtotalgraphb[i]/1000, color=colors[i+1], s=12)
	line3= ax1.scatter(florgraphac[i], dgtotalgraphc[i]/1000, color=colors[i+2], s=12)
	line4= ax1.scatter(florgraphad[i], dgtotalgraphd[i]/1000, color=colors[i+3], s=12)
	line5= ax1.scatter(florgraphae[i], dgtotalgraphe[i]/1000, color=colors[i+4], s=12)
	line6= ax1.scatter(florgraphaf[i], dgtotalgraphf[i]/1000, color=colors[i+5], s=12)

	ax1.plot(linregressXgraph[i], linregressYgraph[i]/1000, '--', color=colors[i])
	ax1.plot(linregressXgraphb[i], linregressYgraphb[i]/1000, '--', color=colors[i+1])
	ax1.plot(linregressXgraphc[i], linregressYgraphc[i]/1000, '--', color=colors[i+2])
	ax1.plot(linregressXgraphd[i], linregressYgraphd[i]/1000, '--', color=colors[i+3])
	ax1.plot(linregressXgraphe[i], linregressYgraphe[i]/1000, '--', color=colors[i+4])
	ax1.plot(linregressXgraphf[i], linregressYgraphf[i]/1000, '--', color=colors[i+5])

	ax1.set_ylabel("$\Delta$G (kJ)")
	ax1.set_xlabel("Temperature (K)")
#This defines where you want the vertial line. You could copy paste this line and add more if desired.
	ax1.axvline(x=343.72, linestyle='--', color='.4')
	ax1.legend((line1, line2, line3, line4, line5, line6), ('CE', 'COC', 'BE', 'NC', 'EME', 'EG'))

#use whichever of these you want to display
print (dgzerototal)
print (dgzerototalb)
print (dgzerototalc)
print (dgzerototald)
print (dgzerototale)
print (dgzerototalf)

plt.tight_layout()
plt.savefig("all_ligands_one_concentration.tiff")





