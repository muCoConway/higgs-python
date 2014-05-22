#! /sw/bin/pythonw

import subroutines as sub
reload(sub)

beam = 125000.0*0.00004/sub.math.sqrt(2.0)
L = 4200.0
ISR_CORR = 0.53
step = 4.07
erange = 9

## All
def all_nocut():
	higgs = [125000,4.07,1.0]
	ex = sub.experiment(1.0,301.4,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$X$","Fit Results: All Events")
	fig.show()
	return fig

def all_totencut():
	higgs = [125000,4.07,1.0]
	ex = sub.experiment(0.52,126.6,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$X$","Fit Results: All Events, $52\%$ Signal Reduction From Energy Cuts")
	fig.show()

## b-bbar
def bbar_nocut():
	higgs = [125000,4.07,0.577]
# b-bbar: 80% b-tag efficiency
	ex = sub.experiment(0.8,57.2,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$b\\bar{b}$","Fit Results: $b\\bar{b}$\nWith 80% Efficiency For Two b-tags")
	fig.show()
	return fig

def bbar_cut():
	higgs = [125000,4.07,0.577]
# b-bbar: 80% b-tag efficiency, 52% cut efficiency
	ex = sub.experiment(0.416,6.3,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$b\\bar{b}$","Fit Results: $b\\bar{b}$\n42% Efficiency With Energy/Event Shape Cuts and b-tag")
	fig.show()
	return fig

# WW*
def ww_nocut():
	higgs = [125000,4.07,0.215]
# WW*: -> l+nu+(any any) = 54%. 100% efficient, minimal bkg
	ex = sub.experiment(0.54,0.001,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$WW^*$","Fit Results: $WW^*$\n54% Decay With $\ell + \\nu_{\ell}$")
	fig.show()
	return fig

# tau+tau-
def tautau_nocut():
	higgs = [125000,4.07,0.063]
#tau+tau-: perfect tag, no cuts
	ex = sub.experiment(1.0,12.8,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$\\tau^+\\tau^-$","Fit Results: $\\tau^+\\tau^-$\nPerfect Tagging")
	fig.show()
	return fig

def tautau_cut():
	higgs = [125000,4.07,0.063]
#tau+tau-: perfect tag, no cuts
	ex = sub.experiment(0.8,5.1,higgs,L,beam,ISR_CORR)
	p = ex.useminos(erange,step,L)
	fig = ex.plot_minos_fit(p,"$\\tau^+\\tau^-$","Fit Results: $\\tau^+\\tau^-$\nPerfect Tagging, Energy, Event Shape Cuts")
	fig.show()
	return fig
