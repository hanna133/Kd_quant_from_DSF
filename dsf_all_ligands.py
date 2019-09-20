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
	data = pd.read_csv("Example.csv", index_col=None)
	coc3=data.loc[data["well"]==cell]

	# fmin=float(coc3["Fl"].min())
	# fmax=float(coc3["Fl"].max())
	def func(x, A2, A1, x0, dx):
		return A2+(A1-A2)/(1+np.exp((x-x0)/dx))

	popt, pcov= curve_fit(func, coc3['temp'], coc3['Fl'],p0=[80000, 80000, 70, .9])

	# print (cell)

	fmax=popt[0]
	fmin=popt[1]
	tm=popt[2]+273
	m=popt[3]
	# print ("tm")
	# print (tm)


	# print ("tm")
	# print (tm)

	# print (Tm)
	# Tmidx=coc3["temp"].iloc[(coc3["temp"]-Tm).abs().argsort()[:1]]
	# print (Tmidx)
	# # Tmidx=Tmidx.index.tolist()
	# fTm=coc3.loc[coc3["temp"]==Tmidx, "Fl"]
	# print (Tmidx)
	# fmax=(fTm-fmin)+fTm
	# print (fmax)
	
	# fig=plt.figure()
	# ax1=fig.add_subplot(111)

	# ax1.plot(coc3['temp'], func(coc3['temp'],*popt), 'g--')
	# ax1.plot(coc3['temp'], coc3['Fl'], 'o')
	# ax1.axvline(x=Tm)
	plt.show()
	# plt.savefig("bfit{cell}.png".format(cell=cell))

	coc3.drop(coc3.tail(10).index, inplace=True)
	coc3=coc3.iloc[10:]
	coc3["temp"]=coc3["temp"]+273
	dgtotal=[]
	index=np.array([1])
	for y in range(1, len(coc3)-1):
		pf=1-((coc3["Fl"].iloc[index]-fmin)/(fmax-fmin))
		index=index+1
		pu=1-pf
		ku=float(pu/pf)
		# ku=float(pu/(1-pu))
		dg=-8.31*coc3["temp"].iloc[index]*np.log(ku)
		dg=float(dg)
		dgtotal.append(dg)
	global dgzero
	dgtotal=np.array(dgtotal)

	coc3=coc3.iloc[ :-2]
	# dgtotal=np.delete(dgtotal, 0)

	slope, intercept, r_value, p_value, std_err = stats.linregress(coc3["temp"], dgtotal)
	tzero=343.63


	dgzero=slope*tzero+intercept

	global linregressX
	global linregressY
	linregressX=range(340,349,1)
	linregressY=[]
	for value in linregressX:
		dg_intercept=slope*value+intercept
		linregressY.append(dg_intercept)
	
	# print (dgzero)
	# print ("dgzero")

	# fig=plt.figure()
	# fig.suptitle(cell)

	# ax1=fig.add_subplot(111)	
	# ax1.scatter(coc3["temp"], dgtotal)
	# ax1.scatter(tzero, dgzero)
	# plt.show()
	# print (r_value**2)
	# plt.savefig("dg{cell}.png".format(cell=cell))
	global florgraph
	florgraph=coc3["temp"]

	return dgzero
	return florgraph
	return tzero
	return dgtotal
	return linregressX
	return linregressY


# celllistA=["A01", "A02","A03", "A04", "A05", "A06", "A07", "A08"]
# celllistB=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]
# celllistC=["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]
# celllistD=["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"]
# celllistE=["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"]
# celllistF=["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"]

celllistA=[ "A08"]
celllistB=[ "B08"]
celllistC=[ "C08"]
celllistD=[ "D08"]
celllistE=[ "E08"]
celllistF=[ "F08"]

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
	ax1.axvline(x=343.72, linestyle='--', color='.4')
	ax1.legend((line1, line2, line3, line4, line5, line6), ('CE', 'COC', 'BE', 'NC', 'EME', 'EG'))
plt.tight_layout()
plt.show()

ddgtotal=[]
# print ("dgzero")
# print (dgzerototal)

index=1
for y in range(1, len(dgzerototal)):
	ddg=dgzerototal[index]-dgzerototal[0]
	# ddg=dgzerototal[0]-dgzerototal[index]
	index +=1
	ddgtotal.append(ddg)
ddgtotal=np.array(ddgtotal)
# print ("ddg")
# print (ddgtotal)

t=343.85 
r=8.31
l=np.array([3.,10.,30.,60.,100.,300.,1000.])/1000000
p=.00000482

kdtotal=[]

index=0

# for y in range(0,len(l)):
# 	kd=-((np.log(8.31)*(310.)+np.log(l[index])*8.31*(310.)-np.log(p*8.31*310))/ddgtotal[index])
# 	kdtotal.append(kd*1000000000)
# 	index+=1
# index=0
# kdquadtotal=[]
# for y in range(0,len(l)):
# 	kdquad=-1*((np.exp((ddgtotal[index])/(r*t)))*(l[index]+(np.exp((ddgtotal[index])/(r*t))*l[index])-p+(np.exp((ddgtotal[index])/(r*t))*p))/(-1+np.exp((2*ddgtotal[index])/(r*t))))
# 	kdquadtotal.append(kdquad*1000000000)
# 	index+=1
# print ("ligand")
# print (kdtotal)
# print ("quadratic")
# print (kdquadtotal)




# # heating ramp rate, first cuouple are faster. 
# # temp for gD, 298, 310, 273
# #protein concentration, just for quadratic
# # reproducibilty of triplicate datasets
# # best results from last three, at midrange concentrations (centered around 100 uM)





##all ligands with all concentrations at tmb
##all of them back to room temp, with the linear regression extrapolated all the way back and vertical line at room temp, tmb, and 37
## top three concentrations at all ligands

##4/19 and 4/10 plus the 4/17 that's already done. 