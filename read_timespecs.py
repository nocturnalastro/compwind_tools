#! /usr/bin/python
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

'''
def timefiles(filename):
    print filename
    dat = [map(float, r.strip().split(" ")) for r in open(filename).read().strip().split("\n")]
    datdat = []
    time = []
    for dd in dat:
        if len(dd) == 1:
            datdat.append([])
            time.append(dd[0])
        else:
            datdat[-1].append(dd)

    datdat = np.vstack(datdat)
    datdat = datdat.reshape(len(time), datdat.shape[0] / len(time), datdat.shape[1])
    return time, datdat

for nk in xrange(0,6):
    dtime, ddat = zip(*[timefiles("spec_extract_time_%i.out" % i) for i in xrange(nk*12,(1+nk)*12)])
    stime, sdat = zip(*[timefiles("spec_scat_extract_time_%i.out" % i) for i in xrange(nk*12,(1+nk)*12)])

    assert np.all((np.diff(np.array(dtime).T) == 0) & (np.diff(np.array(stime).T) == 0)) and np.all((np.array(dtime) - np.array(stime)).T == 0), "not all times are the same something went wrong!"
    dtime = dtime[0]
    # lets avg over all "8" runs
    ddat = np.array(ddat).sum(axis=0)
    sdat = np.array(sdat).sum(axis=0)
    # make binned array
    bin_factor = 10
    bin_ddat = np.zeros((ddat.shape[0], ddat.shape[1] / bin_factor, ddat.shape[2]))
    bin_sdat = np.zeros((sdat.shape[0], sdat.shape[1] / bin_factor, sdat.shape[2]))
    # binning up data
    for n, t in enumerate(dtime):  # each time slice
        for nn in xrange(ddat[n].T.shape[1] / bin_factor):  # we rebing energy and leave the angular stuff
            bin_ddat[n].T[:, nn] = ddat[n].T[:, nn * bin_factor:(nn + 1) * bin_factor].mean(axis=1)
            bin_sdat[n].T[:, nn] = sdat[n].T[:, nn * bin_factor:(nn + 1) * bin_factor].mean(axis=1)
    np.save("times%i"%nk, dtime)
    np.save("timebin_ddat%i"%nk, bin_ddat)
    np.save("timebin_dsat%i"%nk, bin_sdat)

'''
ddat = np.load("dat_avg.npy")
sdat = np.load("sat_avg.npy")
dtime=np.load("times0.npy").tolist()


bin_factor = 25
bin_ddat = np.zeros((ddat.shape[0], ddat.shape[1] / bin_factor, ddat.shape[2]))
bin_sdat = np.zeros((sdat.shape[0], sdat.shape[1] / bin_factor, sdat.shape[2]))
# binning up data
for n, t in enumerate(dtime):  # each time slice
    for nn in xrange(ddat[n].T.shape[1] / bin_factor):  # we rebing energy and leave the angular stuff
        bin_ddat[n].T[:, nn] = ddat[n].T[:, nn * bin_factor:(nn + 1) * bin_factor].mean(axis=1)
        bin_sdat[n].T[:, nn] = sdat[n].T[:, nn * bin_factor:(nn + 1) * bin_factor].mean(axis=1)

binbin_sdat=np.zeros((bin_sdat.shape[0]/5,bin_sdat.shape[1],bin_sdat.shape[2]))
binbin_ddat=np.zeros((bin_ddat.shape[0]/5,bin_ddat.shape[1],bin_ddat.shape[2]))
binbin_time=[]
binbinfactor=5
for i in xrange(bin_ddat.shape[0] / binbinfactor):
    binbin_ddat[i] = bin_ddat[i*binbinfactor:(i + 1) * binbinfactor].mean(axis=0)
    binbin_sdat[i] = bin_sdat[i*binbinfactor:(i + 1) * binbinfactor].mean(axis=0)
    binbin_time.append(np.array(dtime)[i*5:(1 + i) * 5].mean())

dtime=binbin_time
bin_ddat=binbin_ddat
bin_sdat=binbin_sdat

cc=mpl.colors.colorConverter
cols=['#000000', '#FFFF00', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059', '#FFDBE5', '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87', '#5A0007', '#809693', '#FEFFE6', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80', '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100', '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F', '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09', '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66', '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C', '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81', '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00', '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700', '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329', '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C']
cols=[cc.to_rgb(c) for c in cols]
#fig1,ax=plt.subplots(ddat[0].T.shape[0]/4,4,sharex=True)
#fig1,ax=plt.subplots(10,10,sharex=True)
#fig1.subplots_adjust(wspace=0.1,hspace=0.1,right=0.98,left=0.05,top=0.98,bottom=0.05)
#ax=ax.flatten()
#ax[0].set_xlim((0.097,522))

from matplotlib.backends.backend_pdf import PdfPages
pdf_pages = PdfPages('all_times.pdf')
plt.interactive(False)
for n,t in enumerate(dtime):
    fig1,ax=plt.subplots(4,5,sharex=True)
    fig1.subplots_adjust(wspace=0.1,hspace=0.6,right=0.98,left=0.05,top=0.98,bottom=0.05)
    ax=ax.flatten()
    ax[0].set_xlim((0.097,522))
    fig1.set_figwidth(24.0)
    fig1.set_figheight(11.8)
    print t
    for nn in xrange(4,bin_ddat[n].T.shape[0]):
        yy=(bin_ddat[n].T[nn]/bin_ddat[n].T[1])
        ss=(bin_sdat[n].T[nn]/bin_ddat[n].T[1])
        jj=np.hstack([ss,yy,ss-yy])
        ax[nn-4].loglog()
        ax[nn-4].scatter(bin_ddat[n].T[0],yy,marker=".",c="k",s=30,edgecolor="")
        ax[nn-4].scatter(bin_sdat[n].T[0],ss,marker="^",c="r",s=20,edgecolor="")
        ax[nn-4].scatter(bin_sdat[n].T[0],yy-ss,marker="*",c="b",s=20,edgecolor="")
        ax[nn-4].set_ylim([jj.min(),jj.max()])
    fig1.savefig("%i.pdf"%t)
    pdf_pages.savefig(fig1)
    fig1.clf()
    plt.close(fig1)
pdf_pages.close()

'''
from matplotlib.backends.backend_pdf import PdfPages
pdf_pages = PdfPages('all_times_scatter.pdf')
plt.interactive(False)
for n,t in enumerate(dtime):
	fig1,ax=plt.subplots(4,5,sharex=True)
	fig1.subplots_adjust(wspace=0.1,hspace=0.6,right=0.98,left=0.05,top=0.98,bottom=0.05)
	ax=ax.flatten()
	ax[0].set_xlim((0.097,522))
	fig1.set_figwidth(24.0)
	fig1.set_figheight(11.8)
	for nn in xrange(4,ddat[n].T.shape[0]):
		yy=(bin_ddat[n].T[nn]/bin_ddat[n].T[1])
		ss=(bin_sdat[n].T[nn]/bin_sdat[n].T[1])
		ax[nn-4].loglog()
		#ax[nn-4].scatter(bin_ddat[n].T[0],yy,marker=".",c=cols[nn-4],s=5,edgecolor="")
		ax[nn-4].scatter(bin_sdat[n].T[0],ss,marker="^",c="r",s=10,edgecolor="")
		#ax[nn-4].scatter(bin_sdat[n].T[0],yy-ss,marker="*",c=cols[nn-4],s=5,edgecolor="")
		ax[nn-4].set_ylim([yy.min(),yy.max()])
	#fig1.savefig("%i_scat.pdf"%t)
	pdf_pages.savefig(fig1)
	fig1.clf()
	plt.close(fig1)
pdf_pages.close()
'''
