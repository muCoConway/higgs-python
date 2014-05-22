#! /sw/bin/pythonw
import string, math, random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize as opt
from scipy import integrate as integrate
import minuit2

# Subroutines for simulating and fitting Higgs data.
# Estia Echten and Alex Conway

# All energies in MeV

## Fit function things ##

# Class for generating data and fitting to Breit-Wigner convoluted with a Gaussian.
# Stores physics parameters:
# higgs	= [Mh,G,Br]
# Stores given parameters:
# seff	= Higgs signal efficiency as fraction
# bkg	= background cross section
# L		= total luminosity
# beam	= beam spread
# isrcorr = Higgs initial state radiation correction as fraction
# can add alternate fit functions to treat given params as variable
class experiment:
	def __init__(self,
			# Default values
			seff	= 1.0,
			bkg		= 0.0,
			higgs	= [125000,4.07,1.0],
			L		= 4000.0,
			beam	= 125000*0.00004/math.sqrt(2.0),
			isrcorr	= 0.53):
		self.seff = seff
		self.bkg = bkg
		self.L = L
		self.higgs = higgs
		self.beam = beam
		self.isrcorr = isrcorr

	# Generate simulated data
	# returns [ [x], [y], [yerr] ] 
	# erange = generate data within this range (Mh +/- erange)
	# step	= size of step to take (usually == beam)
	def gen_data(self,erange,step,L=-1):
		if (L < 0):
			L = self.L
		higgs = self.higgs
		beam = self.beam
		seff = self.seff
		isrcorr = self.isrcorr
		b = self.bkg
		x = []
		y = []
		yerr = []
		nstep = int(erange/step)
		self.nstep=nstep
		lstep = L/(2.0*nstep)
		self.lstep=lstep
		s = seff*isrcorr*higgs_smear(higgs,beam,higgs[0])
		for i in range(-nstep,nstep+1):
			ecm = higgs[0] + step*i
			s = seff*isrcorr*higgs_smear(higgs,beam,ecm)
			mu = lstep*(s+b)
			N = stats.poisson(mu).rvs()
			x.append(ecm)
			y.append(N)
			if(N>0):
				yerr.append(math.sqrt(N))
			else:
				yerr.append(1)
		out = [ x, y, yerr ]
		self.x = np.array(x)
		self.y = np.array(y)
		self.yerr = np.array(yerr)
		return out

	# Plot bare data with errorbars
	def plot_data(self,data):
		x = data[0]
		y = data[1]
		yerr = data[2]
		plt.errorbar(x,y,yerr,fmt='o')
		plt.show()
	
	# Generate and fit data N times, find std of fit parameters
	def multi_fit(self,N=20,erange=10,step=2.035,lum=4200):
		ms = []
		ws = []
		bs = []
		for i in range(0,N):
			p = self.higgs
			while (p[0] == self.higgs[0]):
				data = self.gen_data(erange,step)
				p,c,chisq,nu,pval = self.do_lsq(data)
			ms.append(p[0]-self.higgs[0])
			ws.append(p[1]-self.higgs[1])
			bs.append(p[2]-self.higgs[2])
		axm = plt.subplot(221)
		plt.hist(ms)
		stdm = np.std(ms)



	# fit data and plot for multiple configurations
	def fit_multi_data(self,eranges=[5,10,15,20],lums=[4200,4200,4200,4200],steps=[3.54,3.54,3.54,3.54]):
		#if (step < 0):
			#step = self.beam
		ms = []
		ws = []
		br = []
		mserr = []
		wserr = []
		brerr = []
		for i in range(len(lums)):
			self.L = lums[i]
			erange = eranges[i]
			step = steps[i]
			p = self.higgs
			while (p[0] == self.higgs[0]):
				data = self.gen_data(erange,step)
				#p,c = self.fit_data(data,False)
				#p,c,chisq,nu,pval = self.do_lsq(data)
			x = data[0]
			y = data[1]
			yerr = data[2]
			axis = plt.subplot(2,2,i+1)
			self.plot_fit_curves(data,axis,p,c,chisq,nu,title="$h^0\\rightarrow b\\bar{b}$")
			ms.append(p[0])
			ws.append(p[1])
			br.append(p[2])
			mserr.append(c[0][0])
			wserr.append(c[1][1])
			brerr.append(c[2][2])
		plt.show()
		return [ms,ws,br],[mserr,wserr,brerr]
			
	def plot_fit_curves(self,data,axis,p=-1,c=np.zeros([3,3]),chisq=-1,nu=-1,title="Fitted Data"):
		if (p[0] < 0):
			p = self.higgs
		L = self.L
		plt.errorbar(data[0],data[1],data[2],fmt='o')
		x_fit = np.linspace(data[0][0],data[0][len(data[0])-1],num=100)
		plt.plot(x_fit,self.convBWG(x_fit,p[0],p[1],p[2]))
		plt.title(title)
		plt.xlabel("$\sqrt{\hat{s}}\ (MeV)$")
		plt.ylabel("Events")
		lbl = "$\mathcal{L}=%d pb^{-1}$\n$\Delta=%.2f\ MeV $" % (self.L,self.beam)
		lbl = lbl + "\nFit results:\n\
$\Delta M_h = %.3f \pm %.3f\ MeV$\n\
$\Gamma_h = %.3f \pm %.3f\ MeV$\n\
$Br(h^0\\rightarrow X) = %.3f \pm %.3f$\n\
$\chi^2/\\nu = %.3f$\
" % (p[0]-self.higgs[0], c[0][0], p[1], c[1][1], p[2], c[2][2], chisq/nu)
		plt.annotate(lbl, [0.6,0.6], xycoords='axes fraction')


	
	# fit the data and optionally plot it
	def fit_data(self,data,show_plot=True):
		x = np.array(data[0])
		y = np.array(data[1])
		yerr = np.array(data[2])

		Mh_guess = self.higgs[0]
		Gh_guess = self.higgs[1]
		Br_guess = self.higgs[2]

		p_guess = [Mh_guess,Gh_guess,Br_guess]
		#p_fit, cov = opt.curve_fit(self.convBWG, x, y, p0=p_guess, sigma=yerr, maxfev=10000)
		p_fit, cov, chisq, nu, pval = self.do_lsq(data)
		if show_plot:
			#x_fit = np.linspace(x[0],x[len(x)-1],num=100)
			#plt.errorbar(x,y,yerr,fmt='o')
			ax = plt.subplot(111)
			#plt.plot(x_fit,self.convBWG(x_fit,p_fit[0],p_fit[1],p_fit[2]))
			#plt.show()
			self.plot_fit_curves(data,ax,p_fit,cov,title="Data Sim and Fit for $h^0\\rightarrow b\\bar{b}$")
			plt.show()
		return p_fit, cov

	def fit_vs_lum(self,erange,step,lums=[1000,2000,3000,4000]):
		x = lums
		m = []
		w = []
		br = []
		merr = []
		werr = []
		brerr = []
		for i in range(len(x)):
			p = self.higgs
			self.L = lums[i]
			while (p[0] == self.higgs[0]):
				data = self.gen_data(erange,step)
				p,c = self.fit_data(data,False)
				#p,c,chisq,nu,pval = self.do_lsq(data)
			m.append(p[0]-self.higgs[0])
			w.append(p[1])
			br.append(p[2])
			merr.append(c[0][0])
			werr.append(c[1][1])
			brerr.append(c[2][2])
		xmin = min(lums)-500
		xmax = max(lums)+500
		plt.subplot(321)
		plt.ylabel("$\\Delta M_h (MeV)$")
		plt.errorbar(x,m,yerr=merr)
		plt.plot([xmin,xmax],[0,0],ls='dashed',linewidth=2)
		plt.xlim(xmin=xmin,xmax=xmax)
		plt.subplot(323)
		plt.ylabel("$\\Gamma_h (MeV)$")
		plt.errorbar(x,w,yerr=werr)
		plt.plot([xmin,xmax],[self.higgs[1],self.higgs[1]],ls='dashed',linewidth=2)
		plt.xlim(xmin=xmin,xmax=xmax)
		plt.subplot(325)
		plt.ylabel("$Br(h^0\\rightarrow X)$")
		plt.errorbar(x,br,yerr=brerr)
		plt.plot([xmin,xmax],[self.higgs[2],self.higgs[2]],ls='dashed',linewidth=2)
		plt.xlabel("Integrated Luminsoity $(pb^{-1})$")
		plt.xlim(xmin=xmin,xmax=xmax)
		plt.subplot(322)
		zeros = np.zeros(len(x))
		plt.ylabel("$\\Delta M_h (MeV)$")
		plt.errorbar(x,zeros,yerr=merr)
		plt.xlim(xmin=xmin,xmax=xmax)
		plt.subplot(324)
		plt.ylabel("$\\Gamma_h (MeV)$")
		plt.errorbar(x,zeros,yerr=werr)
		plt.xlim(xmin=xmin,xmax=xmax)
		plt.subplot(326)
		plt.ylabel("$Br(h^0\\rightarrow X)$")
		plt.errorbar(x,zeros,yerr=brerr)
		plt.xlabel("Integrated Luminsoity $(pb^{-1})$")
		plt.xlim(xmin=xmin,xmax=xmax)
		plt.show()


	# return convoluted Breit-Wigner for use in fitting.
	# x = array of ecm's
	# parameters: mass, width, branching fraction
	# constants: self.{L,beam,bkg,seff,isrcorr}
	def convBWG(self, x, M, G, bf, 
			b=-1, lstep=-1, seff=-1, beam=-1, isrcorr=-1):
		if (b < 0):
			b = self.bkg
		if (lstep < 0):
			lstep = self.lstep
		if (seff < 0):
			seff = self.seff
		if (beam < 0):
			beam = self.beam
		if (isrcorr < 0):
			isrcorr = self.isrcorr
		out = []
		for i in range(len(x)):
			s = higgs_smear([M,G,bf],beam,x[i])
			s = s*seff*isrcorr
			out.append(lstep*(s+b))
		return out

	# More customized fitting:
	# fit function is same as convBWG
	def fitfunc(self, p, x):
		M = p[0]
		G = p[1]
		bf = p[2]
		b = self.bkg
		lstep = self.lstep
		seff = self.seff
		beam = self.beam
		isrcorr = self.isrcorr
		out = []
		for i in range(len(x)):
			s = higgs_smear([M,G,bf],beam,x[i])
			s = s*seff*isrcorr
			out.append(lstep*(s+b))
		return out
	# Perform least squares fit on data
	def do_lsq(self, data):
		x = np.array(data[0])
		y = np.array(data[1])
		yerr = np.array(data[2])
		p0 = self.higgs
		p,c,info,msg,ier = opt.leastsq(self.errfunc, p0, args=(x,y,yerr), full_output=1)
		#c = c*(math.sqrt(sum(info['fvec']**2)))
		print(p)
		print(c)
		#c = np.array(c)
		#c = c**0.5
		#print(c)
		chisq = sum(info['fvec']*info['fvec'])
		nu = len(x) - (len(p)+1)
		chisq_red = chisq/nu
		pval = stats.chi2.sf(chisq,nu)
		print(chisq)
		print(nu)
		print(chisq_red)
		print(pval)
		print(info['nfev'])
		print(msg)
		print(ier)
		return p,c,chisq,nu,pval#,info,msg,ier
	# weight the error by uncertainty
	def errfunc(self, p, x, y, yerr):
		fx = self.fitfunc(p,x)
		return (y-fx)/yerr
	def chisquare(self,M,G,B):
		x = self.x
		y = self.y
		yerr = self.yerr
		chi = self.errfunc([M,G,B],x,y,yerr)
		return sum(chi**2)

	# simulate and fit data with migrad, then calc errors with minos
	def useminos(self,erange=10,step=4.07,lum=4200):
		data = self.gen_data(erange,step,lum)
		m = minuit2.Minuit2(self.chisquare)
		m.values = {
				'M': self.higgs[0],
				'G': self.higgs[1],
				'B': self.higgs[2] }
		m.errors = {
				'M': 0.1,
				'G': 0.1,
				'B': 0.1 }
		#limits slow calculation
		#m.limits = {
				#'M': (124999.0,125001.0),
				#'G': (3.5,4.5),
				#'B': (0.4,0.7)}
		# scale factor for 1-sigma errors
		# (1 for chisq, 0.5 for -log(likelihood)
		m.up = 1
		m.printMode = 1
		m.migrad()
		m.minos()
		print("minos: ncalls, merrors:, m.args")
		print(m.ncalls)
		print(m.values)
		print(m.merrors)
		mout = [m.values['M'],m.merrors['M',-1],m.merrors['M',1]]
		gout = [m.values['G'],m.merrors['G',-1],m.merrors['G',1]]
		bout = [m.values['B'],m.merrors['B',-1],m.merrors['B',1]]
		return [mout,gout,bout]

	def plot_minos_fit(self,p,decay="X",title="Fit Results",erange=9,step=4.07,lum=4200):
		fig = plt.figure(figsize=(8,6))
		plt.errorbar(self.x,self.y,self.yerr,fmt='o')
		M = p[0][0]
		G = p[1][0]
		B = p[2][0]
		dMl = p[0][1]
		dMu = p[0][2]
		dGl = p[1][1]
		dGu = p[1][2]
		dBl = p[2][1]
		dBu = p[2][2]
		x_fit = np.linspace(min(self.x),max(self.x),num=100)
		plt.plot(x_fit,self.convBWG(x_fit,M,G,B))
		plt.xlabel("$\sqrt{\hat{s}} (MeV)$",fontsize=16)
		plt.ticklabel_format(useOffset=False)
		plt.ylabel("Counts",fontsize=16)
		plt.title(title,fontsize=16)
		lbl1 = "Input:\n$\mathcal{L}=%d pb^{-1}$\n$\Delta=%.3f\ MeV$\n$\delta\sqrt{\hat{s}} = %.3f MeV$" % (lum,step,self.beam)
		lbl1 = lbl1 + "\n$M_h = 125.0 GeV$\n$\Gamma_h = 4.07 MeV$\n$Br(h^0\\rightarrow$%s$) = %.3f$" % (decay, self.higgs[2])
		lbl1 = lbl1 + "\n$\sigma_{bkg} = %.2f pb^{-1}$" % (self.bkg)
		lbl2 = "\nFit results:\n"
		lbl2 = lbl2 + "$\Delta M_h = %.3f_{-%.3f}^{+%.3f}\ MeV$\n" % (M-self.higgs[0], -1*dMl, dMu)
		lbl2 = lbl2 + "$\Gamma_h = %.3f_{-%.3f}^{+%.3f} \ MeV$\n" % (G, -1*dGl, dGu)
		lbl2 = lbl2 + "$Br(h^0\\rightarrow$%s$) = %.3f_{-%.3f}^{+%.3f}$\n" % (decay, B, -1*dBl, dBu)
		plt.annotate(lbl1, [0.1,0.6], xycoords='axes fraction',fontsize=15)
		plt.annotate(lbl2, [0.7,0.6], xycoords='axes fraction',fontsize=15)
		return plt
	

	def plot_minos(self,eranges=[9,9,9,9],steps=[4.07,4.07,4.07,4.07],lum=4200):
		for i in range(0,len(eranges)):
			p= self.useminos(eranges[i],steps[i],lum)
			plt.subplot(2,2,i+1)
			plt.errorbar(self.x,self.y,self.yerr,fmt='o')
			M = p[0][0]
			G = p[1][0]
			B = p[2][0]
			dMl = p[0][1]
			dMu = p[0][2]
			dGl = p[1][1]
			dGu = p[1][2]
			dBl = p[2][1]
			dBu = p[2][2]
			x_fit = np.linspace(self.x[0],self.x[len(self.x)-1],num=100)
			plt.plot(x_fit,self.convBWG(x_fit,M,G,B))
			lbl = "$\mathcal{L}=%d pb^{-1}$\n$\Delta=%.3f\ MeV $" % (lum,steps[i])
			lbl = lbl + "\nFit results:\n"
			lbl = lbl + "$\Delta M_h = %.3f_{-%.3f}^{+%.3f}\ MeV$\n" % (M-self.higgs[0], -1*dMl, dMu)
			lbl = lbl + "$\Gamma_h = %.3f_{-%.3f}^{+%.3f} \ MeV$\n" % (G, -1*dGl, dGu)
			lbl = lbl + "$Br(h^0\\rightarrow X) = %.3f_{-%.3f}^{+%.3f}$\n" % (B, -1*dBl, dBu)
			plt.annotate(lbl, [0.7,0.6], xycoords='axes fraction')
		plt.show()

	def red_chisq(self,M,G,B):
		y_fit = np.array(self.convBWG(self.x,M,G,B))
		chisq = sum(((self.y - y_fit)/self.yerr)**2)
		dof = len(self.x) - 1 - 3
		return chisq/dof

	
	def plot_contours(self,recalc=True,erange=10,step=2.035,lum=4200):
		if (recalc):
			data = self.gen_data(erange,step,lum)
			m = minuit2.Minuit2(self.chisquare)
			m.values = {
					'M': self.higgs[0],
					'G': self.higgs[1],
					'B': self.higgs[2] }
			m.up = 1
			m.printMode = 1
			m.migrad()
			m.minos()
			self.minuit = m
			self.contour_mg = np.array(m.contour('M','G',1))
			self.contour_mb = np.array(m.contour('M','B',1))
			self.contour_gb = np.array(m.contour('G','B',1))
		m = self.minuit
		contour_mg = self.contour_mg
		contour_mb = self.contour_mb
		contour_gb = self.contour_gb
		plt.subplot(2,2,1)
		plt.plot(contour_mg[:,0],contour_mg[:,1])
		plt.xlabel("Mass")
		plt.ylabel("Gamma")
		plt.subplot(2,2,2)
		plt.plot(contour_mb[:,0],contour_mb[:,1])
		plt.xlabel("Mass")
		plt.ylabel("Branching Ratio")
		plt.subplot(2,2,3)
		plt.plot(contour_gb[:,0],contour_gb[:,1])
		plt.xlabel("Gamma")
		plt.ylabel("Branching Ratio")
		plt.subplot(2,2,4)
		plt.errorbar(self.x,self.y,self.yerr,fmt='o')
		M = m.values['M']
		G = m.values['G']
		B = m.values['B']
		dMl = m.merrors['M',-1]
		dMu = m.merrors['M',1]
		dGl = m.merrors['G',-1]
		dGu = m.merrors['G',1]
		dBl = m.merrors['B',-1]
		dBu = m.merrors['B',1]
		x = np.linspace(min(self.x),max(self.x),num=100)
		y = self.convBWG(x,M,G,B)
		lbl = "$\mathcal{L}=%d pb^{-1}$\n$\Delta=%.3f\ MeV $" % (lum,step)
		lbl = lbl + "\nFit results:\n"
		lbl = lbl + "$\Delta M_h = %.3f_{-%.3f}^{+%.3f}\ MeV$\n" % (M-self.higgs[0], -1*dMl, dMu)
		lbl = lbl + "$\Gamma_h = %.3f_{-%.3f}^{+%.3f} \ MeV$\n" % (G, -1*dGl, dGu)
		lbl = lbl + "$Br(h^0\\rightarrow X) = %.3f_{-%.3f}^{+%.3f}$" % (B, -1*dBl, dBu)
		plt.plot(x,y)
		plt.xlabel("$\sqrt{s}(MeV)$")
		plt.ylabel("Counts")
		plt.annotate(lbl, [0.7,0.6], xycoords='axes fraction')
		plt.show()

	
	def plot_fmin(self,erange=10,step=2.035,lum=4200):
		p = self.useminos(erange,step,lum)
		M = p[0][0]
		G = p[1][0]
		B = p[2][0]
		dMl = p[0][1]
		dMu = p[0][2]
		dGl = p[1][1]
		dGu = p[1][2]
		dBl = p[2][1]
		dBu = p[2][2]
		plt.subplot(2,2,1)
		x = np.linspace(dMl*2+M,dMu*2+M,50)
		y = []
		for i in range(0,len(x)):
			y.append(self.chisquare(x[i],G,B))
		plt.plot(x,y)
		plt.plot([M,M],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.plot([M+dMl,M+dMl],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.plot([M+dMu,M+dMu],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.subplot(2,2,2)
		x = np.linspace(dGl*2+G,dGu*2+G,50)
		y = []
		for i in range(0,len(x)):
			y.append(self.chisquare(M,x[i],B))
		plt.plot(x,y)
		plt.plot([G,G],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.plot([G+dGl,G+dGl],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.plot([G+dGu,G+dGu],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.subplot(2,2,3)
		x = np.linspace(dBl*2+B,dBu*2+B,50)
		y = []
		for i in range(0,len(x)):
			y.append(self.chisquare(M,G,x[i]))
		plt.plot(x,y)
		plt.plot([B,B],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.plot([B+dBl,B+dBl],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.plot([B+dBu,B+dBu],[min(y),max(y)],ls='dashed',linewidth=2)
		plt.subplot(2,2,4)
		x = np.linspace(dMl*2+M,dMu*2+M,50)
		y = np.linspace(dGl*2+G,dGu*2+G,50)
		zs = []
		for m in x:
			z = []
			for g in y:
				c2 = self.chisquare(m,g,B)
				z.append(c2)
			zs.append(z)
		v = [self.chisquare(M,G,B),self.chisquare(M+0.5*dMu,G,B),self.chisquare(M+dMu,G,B),self.chisquare(M+1.5*dMu,G,B),self.chisquare(M+2*dMu,G,B),self.chisquare(M,G,B),self.chisquare(M,G+0.5*dGu,B),self.chisquare(M,G+dGu,B),self.chisquare(M,G+1.5*dGu,B),self.chisquare(M,G+2*dGu,B)]
		plt.contour(x,y,zs,v)
		plt.show()
		
					

	def test_minos(self,recalc=True,N=10,erange=9,step=4.07,lum=4200):
		if (recalc):
			ms = []
			ws = []
			bs = []
			pulls = []
			for i in range(0,N):
				p = self.useminos(erange,step,lum)
				ms.append(p[0])
				ws.append(p[1])
				bs.append(p[2])
				y_fit = self.convBWG(self.x,p[0][0],p[1][0],p[2][0])
				y_tru = self.convBWG(self.x,self.higgs[0],self.higgs[1],self.higgs[2])
				for i in range(len(self.x)):
					pulls.append((y_tru[i]-y_fit[i])/math.sqrt(y_fit[i]))
			ms = np.array(ms)
			ws = np.array(ws)
			bs = np.array(bs)
			mvs = ms[:,0]
			wvs = ws[:,0]
			bvs = bs[:,0]
			mes = [-1*ms[:,1],ms[:,2]]
			wes = [-1*ws[:,1],ws[:,2]]
			bes = [-1*bs[:,1],bs[:,2]]
			self.mvs = mvs
			self.wvs = wvs
			self.bvs = bvs
			self.mes = mes
			self.wes = wes
			self.bes = bes
			self.pulls = pulls
		mvs = self.mvs
		wvs = self.wvs
		bvs = self.bvs
		mes = self.mes
		wes = self.wes
		bes = self.bes
		pulls = self.pulls
		plt.figure(1)
		plt.subplot(221)
		plt.errorbar(range(0,len(mvs)),mvs-self.higgs[0],yerr=mes,fmt='o')
		plt.plot([-1,len(mvs)],[0,0],ls='dashed',linewidth=2)
		plt.ylabel("Fitted Mass (MeV)")
		plt.subplot(222)
		plt.errorbar(range(0,len(wvs)),wvs-self.higgs[1],yerr=wes,fmt='o')
		plt.plot([-1,len(mvs)],[0,0],ls='dashed',linewidth=2)
		plt.ylabel("Fitted Width (MeV)")
		plt.subplot(223)
		plt.errorbar(range(0,len(bvs)),bvs-self.higgs[2],yerr=bes,fmt='o')
		plt.plot([-1,len(mvs)],[0,0],ls='dashed',linewidth=2)
		plt.ylabel("Fitted Branching Ratio")
		#test = np.mean(np.std(bvs,axis=0)/np.mean(bes,axis=0))
		#test = self.fitparamchisq(bvs-self.higgs[2],bes)
		#lbl = "test: %.3f" % (test)
		#plt.annotate(lbl, [0.6,0.6], xycoords='axes fraction')
		plt.subplot(224)
		plt.hist(pulls)
		x = np.linspace(min(pulls),max(pulls),num=100)
		y = math.sqrt(2)*len(pulls)*stats.norm.pdf(x)/math.sqrt(10)
		plt.plot(x,y)
		print(stats.normaltest(pulls))
		plt.xlabel("Normalized Residuals")
		plt.figure(2)
		plt.subplot(221)
		plt.xlabel("Mass Fit Pulls")
		pull_m = self.fitparampull(mvs-self.higgs[0],mes)
		k2,p = stats.normaltest(pull_m)
		print("\nmass k2 and p:")
		print(k2)
		print(p)
		plt.hist(pull_m)
		x = np.linspace(min(pull_m),max(pull_m),num=100)
		y = math.sqrt(2)*len(pull_m)*stats.norm.pdf(x)/math.sqrt(10)
		plt.plot(x,y)
		plt.subplot(222)
		plt.xlabel("Width Fit Pulls")
		pull_w = self.fitparampull(wvs-self.higgs[1],wes)
		k2,p = stats.normaltest(pull_w)
		print("\nwidth k2 and p:")
		print(k2)
		print(p)
		plt.hist(pull_w)
		x = np.linspace(min(pull_w),max(pull_w),num=100)
		y = math.sqrt(2)*len(pull_w)*stats.norm.pdf(x)/math.sqrt(10)
		plt.plot(x,y)
		plt.subplot(223)
		plt.xlabel("Branching Ratio Fit Pulls")
		pull_b = self.fitparampull(bvs-self.higgs[2],bes)
		k2,p = stats.normaltest(pull_b)
		print("\nbranching. k2 and p:")
		print(k2)
		print(p)
		plt.hist(pull_b)
		x = np.linspace(min(pull_b),max(pull_b),num=100)
		y = math.sqrt(2)*len(pull_b)*stats.norm.pdf(x)/math.sqrt(10)
		plt.plot(x,y)

		plt.show()

	def fitparampull(self,vs,es):
		rv=[]
		for i in range(len(vs)):
			if (vs[i] < 0):
				rv.append((vs[i])/(es[0][i]))
			else:
				rv.append((vs[i])/(es[1][i]))
		return rv



	
## Higgs cross section calculations ##

# Calculate Higgs Breit-Wigner cross section at a given energy.
# higgs_in = [M_h, G_h, Br]
# Output in... barns?
def higgs_xsect(higgs_in, ecm):
	xsect = 0.0
	convert = 0.3894
	Bfmu = 0.000209
	mH = higgs_in[0]/1000.0
	gH = higgs_in[1]/1000.0
	Bf = higgs_in[2]
	ecm = ecm/1000.0
	shat = ecm*ecm
	num = 4*np.pi*gH*gH*Bf*Bfmu
	denom = (shat - mH*mH)*(shat - mH*mH) + gH*gH*mH*mH
	xsect = convert*num/denom
	return xsect

# Smear Higgs cross section by convolution with Gaussian beam.
# beam = beam gaussian spread in MeV
# Output in pb
def higgs_smear(higgs_in, beam, ecm):
	mH = higgs_in[0]
	gH = higgs_in[1]
	Bf = higgs_in[2]
	smear0 = np.sqrt(2*np.pi)*beam
	smear1 = 2*beam*beam
	sum = 0.0
	#nsteps = 5000
	nsteps = 1000
	use_limit = 10.0
	erange = use_limit*beam
	delta = 0.5*erange/nsteps
	for i in range(-nsteps,nsteps+1):
		ev = ecm + i*delta
		smear2 = np.exp(-(ev-ecm)*(ev-ecm)/smear1)/smear0
		val = higgs_xsect(higgs_in,ev)
		sum += delta*smear2*val
	sum = sum*1.0e+9
	return sum

