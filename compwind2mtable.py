#! /usr/bin/python
import numpy as np
import os,sys
from itertools import product
import pyfits as fits
# physical constants SI units
G = 6.67e-11
MSUN = 1.98892e30
C = 2.998e8
MPROTON = 1.67262e-27
SIGMA_T = 6.6524e-29
YEAR = 365.24*24*3600.
EPSILON = 0.06  # radiative efficiency

if len(sys.argv[1:])>0:
	model_name=sys.argv[1]
else:
	import time
	model_name="dw_"+time.strftime("%d%b%y")


def return_vals(f):
	return [ np.float(v.strip()) if len(v.strip().split(" "))==1 else [np.float(x) for x in v.split(" ") if len(x)>0] for v in f.read().split("\n") if len(v)>0]

def get_spec(filename):
	with open(filename) as f:
		i=f.read().split("\n")[:init_size]
		i=[np.array([float(z) for z in q.split(" ") if len(z)>0 ]) for q in i if len(q)>0]
		d=np.vstack(i).T
		return d

redshiftkeyval = 1
additivemodel = 0
init_size = 10000
npar = 11  # numer of parameters not including mu
tol = 5.e-6  # tolerance for matching of parameter values to the grid


parval=[[] for _ in xrange(npar)]
nparm=[]
input_files_set=set(["input_extract.txt","input_model.txt","input_spec.txt","input.txt"])
flname="testing.txt"
dirs=[x.replace("\n","") for x in open(flname).readlines() if len(x.replace("\n",""))>0]

spec_tot_mu=[[] for _ in dirs]
spec_sca_mu=[[] for _ in dirs]
spec_dir_mu=[[] for _ in dirs]

grid=[[] for _ in xrange(npar)]
print "dir, Temp, Dval, Rin, Rout/Rin, Mdot, vinf, vscaler, clump, Lx, abund"
for model_num,dr in enumerate(dirs):
	filelist=os.listdir(dr)
	if input_files_set.issubset(filelist):
		with open(dr+"/input.txt") as f:
			input_values=return_vals(f)
			#print input_values
			M_BH,oboundary,=input_values[1][:-1]
			Temp,Lx,Lbol=input_values[3]
			Rem=input_values[10]

			RG = G*M_BH*MSUN/C/C #calc RG for scaling!!!
			RG = RG*100. # convert to cm
			Ledd = 4.*np.pi*G*M_BH*MSUN*MPROTON*C/SIGMA_T #calc Ledd for scaling!!!
			Ledd = Ledd*1.e7 #convert to erg/sec
			Lx=100*np.power(10.0,Lx)
			Lx/=Ledd

			M_BH=np.log10(M_BH) #convert M_BH for use later
			oboundary/=RG
			Rem/=RG
			Temp=np.log10(abs(Temp))
			del(input_values)

		with open(dr+"/input_model.txt") as f:
			Rin,Rout,Dval,Mdot,Kval,v0,vinf,vscaler,vbeta,clump=return_vals(f)
			Dval /= Rin
			Rin/=RG
			Rout/=Rin*RG
	  		Mdot = (EPSILON*Mdot*MSUN*C*C*1.e7)/YEAR/Ledd
	  		vscaler/=Rout*RG


		with open(dr+"/input_extract.txt") as f:
			nmuparam=int(f.readline().replace("\n","").strip())

	  	with open(dr+"/input_spec.txt") as f:
			nspecparam=f.readline().replace("\n","").strip()


	## todo look up acutally look up abund!
	abund=1
	## todo
	parname=["Gamma","logT","D","R_in","R_out","Mdot","Vinf","VscaleR","Clumping","% L2-10", "Fe-grp abund"]

	parval[1].append(Temp)
	parval[2].append(Dval)
	parval[3].append(Rin)
	parval[4].append(Rout)
	parval[5].append(Mdot)
	parval[6].append(vinf)
	parval[7].append(vscaler)
	parval[8].append(clump)
	parval[9].append(Lx)
	parval[10].append(abund)
	#move these
	npspec_set=set(["spec_bin_extract.out.np","spec_bin_scat.out.np","spec_bin_direct.out.np"])

	if npspec_set.issubset(filelist):
		spec_tot=np.load(dr+"/spec_bin_extract.out.np")
		spec_sca=np.load(dr+"/spec_bin_scat.out.np")
		spec_dir=np.load(dr+"/spec_bin_direct.out.np")
	else:
		spec_tot=get_spec(dr+"/spec_bin_extract.out")
		spec_sca=get_spec(dr+"/spec_bin_scat.out")
		spec_dir=get_spec(dr+"/spec_bin_direct.out")

	for i in xrange(nmuparam):
		spec_tot_mu[model_num].append(spec_tot[4+i]/spec_tot[1])
		spec_sca_mu[model_num].append(spec_sca[4+i]/spec_sca[1])
		spec_dir_mu[model_num].append(spec_dir[4+i]/spec_dir[1])

	gamma=-round((np.log10(spec_tot[1][1])-np.log10(spec_tot[1][-1]))/(np.log10(spec_tot[0][1])-np.log10(spec_tot[0][-1])),5)
	parval[0].append(gamma)
	print dr,gamma, Temp, Dval, Rin, Rout, Mdot, vinf, vscaler, clump, Lx, abund


for par in xrange(npar):
	grid[par]=sorted(np.unique(parval[par]))
	nparm.append(len(grid[par]))


print "Read %d models." % model_num
with open(dr+"/input_extract.txt") as f:
	muvals=sorted(map(lambda x: round(x,3),np.cos(np.deg2rad([np.float(x) for x in f.read().split("\n")[1:] if len(x.strip())>0]))))


print " grid parameter values:"
for par in xrange(npar):
	grid[par].sort()
	print " parameter %d: %s: " % (par+1,parname[par])
	print grid[par]

print " parameter %d: %s: " % (npar+1,"mu")
print muvals

nparm.append(len(muvals))

tparval=[np.array(pv) for pv in parval]
def compval(arr,val):
	return set(np.where(abs(arr-val)<=tol)[0])

tot_array=[]
sca_array=[]
dir_array=[]
gridval=[]



identified=[0 for _ in xrange(model_num+1)]

for gamma,temp,dval,rin,rout,mdot,vinf,vscaler,clump,Lx,abund in product(*grid): #gets all combinations of parameters in order
	try:
		tmodel=	(compval(parval[0],gamma) & compval(parval[1],temp) & compval(parval[2],dval) & compval(parval[3],rin) & compval(parval[4],rout) & compval(parval[5],mdot) & compval(parval[6],vinf) & compval(parval[7],vscaler) & compval(parval[8],clump) & compval(parval[9],Lx) & compval(parval[10],abund)).pop()
		identified[tmodel]+=1
	except KeyError:
		raise KeyError("error, not all grid points have a spectrum \nMissing specturm for values:\n%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (gamma,temp,dval,rin,rout,mdot,vinf,vscaler,clump,Lx,abund))
	for mu,muval in enumerate(muvals):
		tot_array.append(spec_tot_mu[tmodel][mu])
		sca_array.append(spec_sca_mu[tmodel][mu])
		dir_array.append(spec_dir_mu[tmodel][mu])
		val=[pv[tmodel] for i,pv in enumerate(parval) if nparm[i]>1 ]+[muval]+[0]
		gridval.append(val)

if not np.all(np.asarray(identified)==1):
	raise KeyError("Error in placing input spectrum into grid here are how many times each grid was used (should be = 1):\n"+",".join(map(str,identified)))




# create the array of low and high energy values of each energy bin
elo=[0]
ehi=[]
energy=spec_tot[0]
for j,e in enumerate(energy):
	if j<len(energy)-1:
		ehi.append((e+energy[j+1])/2)
		if j>0:
			elo.append(ehi[j])

elo= [elo[1] - (energy[1]-energy[0])]+elo
ehi.append(ehi[-1] + (energy[-1]-energy[-2]))

elow = elo[0]
ehigh = ehi[j-1]

#now start the FITS output
#parameter values for first table
#find max number grid values needed and write into fits arrays
pri=fits.PrimaryHDU()
pri.header['BITPIX']=16
for key,val,comment in [('MODLNAME',model_name[:12],'first 12 characters of model name'),('MODLUNIT','','model unit'),('REDSHIFT',True,'is redshift a parameter?'),('HDUCLASS','OGIP','format conforms to OGIP standard'),('DUDOC'  ,'OGIP/92-009','document defining format'),('HDUCLAS1','XSPEC TABLE MODEL','model spectra for XSPEC'),('HDUVERS1','1.0.0','version of format'),('AUTHOR','Lance Miller','author of mtable creation code'),('ADDMODEL',False,'Additive model? no')]:
	pri.header[key]=(val,comment)


method=1

parstring=["photon index","log T_e","D","R_in","R_out","M_dot", "Vinf/Vesc", "VscaleR/R_out","Clumping parameter", "% 2-10keV Lum", "[Fe/H]","mu"]


largestnpram=max([len(np.unique(pp)) for pp in parval+[muvals]])

print "!!!!!!!!!!!!!!",largestnpram,np.argmax([len(np.unique(pp)) for pp in parval+[muvals]])
ttype1 = [("NAME","12A"),("METHOD","J"),("INITIAL","D"),("DELTA","D"),("MINIMUM","D"),("BOTTOM","D"),("TOP","D"),("MAXIMUM","D"),("NUMBVALS","J"),("VALUE","%iD" % largestnpram)]  #names of columns

rnddecimal=10**6 #so ceil and floor can be used to round values

initial=[np.ceil(x[0]*rnddecimal)/rnddecimal for x in grid if len(x)>1]+[np.round(muvals[0],3)] #initial value
maximum=[np.floor(x[-1]*rnddecimal)/rnddecimal for x in grid if len(x)>1]+[np.round(muvals[-1],3)] #maximum value
delta=[x[1]-x[0] for x in grid if len(x)>1]+[np.round(muvals[0]+muvals[1])] #increment value

ttvals=[[parstring[n] for n,m in enumerate(nparm)  if m>1],[method for n in nparm  if n>1],initial,delta,initial,initial,maximum,maximum,[n for n in nparm  if n>1],[np.unique(p).tolist()+[0 for _ in xrange(largestnpram-np.unique(p).shape[0])] for p in parval if len(np.unique(p))>1 ]+[muvals+[0 for _ in xrange(largestnpram-np.unique(muvals).shape[0])]]]

para_cols=[]
for i,(colname,coltype) in enumerate(ttype1):
	print colname,coltype,ttvals[i]
	para_cols.append(fits.Column(colname,coltype,array=ttvals[i]))

parameters_table=fits.BinTableHDU.from_columns(para_cols)

for key,val,comment in [('EXTNAME','PARAMETERS','name of this binary table extension'),('HDUCLASS','OGIP','format conforms to OGIP standard'),('HDUDOC','OGIP/92-009','document defining format'),('HDUCLAS1','XSPEC TABLE MODEL' ,'model spectra for XSPEC'),('HDUVERS1','1.0.0','version of format'),('GAMMA',gamma,'photon index'),('LOGT',temp,'log T_e'),('D',Dval,'D parameter'),('R_IN',Rin,'R_in/rg'),('R_OUT',Rout,'R_out/R_in'),('VINF',vinf,' Vinf/Vesc'),('VSCALER',vscaler,' VscaleR/R_out'),('CLUMPING',clump,'Clumping parameter'),('HIERARCH Fe-grp abund',abund,' [Fe/H]'),('NINTPARM',len([x for x in nparm if x>1]),'number of interpolated parameters'),('NADDPARM',0,'number of additional parameters'),('ELOW',elow,'energy band low end (eV)'),('EHIGH',ehigh,'energy band high end (eV)'),('M_BH',M_BH,'black hole mass/solar'),('LUM',Lx,'L_x/L_Edd'),('OBOUND',oboundary,'outer boundary/rg '),('REM',Rem,'size of emission region/rg'),('V0',0.,'launch velocity'),('VBETA',1.,'velocity law beta'),('KVAL',-1.,' mass-loss exponent')]:
	parameters_table.header[key]=(val,comment)

en_cols=[]
en_cols.append(fits.Column("ENERG_LO","D","keV",array=elo))
en_cols.append(fits.Column("ENERG_HI","D","keV",array=ehi))

energies_table=fits.BinTableHDU.from_columns(en_cols)

for key,val,comment in [('EXTNAME','ENERGIES','name of this binary table extension'),('HDUCLASS','OGIP','format conforms to OGIP standard'),('HDUDOC','OGIP/92-009','document defining format'),('HDUCLAS1','XSPEC TABLE MODEL','model spectra for XSPEC'),('HDUCLAS2','ENERGIES','expression containing energy info'),('HDUVERS1','1.0.0','version of format')]:
	energies_table.header[key]=(val,comment)



spec_tot_col=[fits.Column('PARAMVAL','%iD' % len(gridval[0]),"",array=gridval)]
spec_dir_col=[fits.Column('PARAMVAL','%iD' % len(gridval[0]),"",array=gridval)]
spec_sca_col=[fits.Column('PARAMVAL','%iD' % len(gridval[0]),"",array=gridval)]
spec_tot_col.append(fits.Column('INTPSPEC','%iD' % len(elo),"",array=tot_array))
spec_dir_col.append(fits.Column('INTPSPEC','%iD' % len(elo),"",array=dir_array))
spec_sca_col.append(fits.Column('INTPSPEC','%iD' % len(elo),"",array=sca_array))

spec_tot_table=fits.BinTableHDU.from_columns(spec_tot_col)
spec_dir_table=fits.BinTableHDU.from_columns(spec_dir_col)
spec_sca_table=fits.BinTableHDU.from_columns(spec_sca_col)

for t in  [spec_tot_table,spec_dir_table,spec_sca_table]:
	t.header['EXTNAME']=('SPECTRA','name of this binary table extension')


tot_hdu=fits.HDUList([pri,parameters_table,energies_table,spec_tot_table])
dir_hdu=fits.HDUList([pri,parameters_table,energies_table,spec_dir_table])
sca_hdu=fits.HDUList([pri,parameters_table,energies_table,spec_sca_table])


tot_hdu.writeto('%s_total.fits' % model_name)
dir_hdu.writeto('%s_direct.fits' % model_name)
sca_hdu.writeto('%s_scatter.fits' % model_name)
