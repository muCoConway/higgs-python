#! /sw/bin/pythonw
import subroutines as sub

# Make plots of higgs fit data.

fit = sub.mybwgfit()

def plot_data(
		erange	= 30,
		step	= fit.beam,
		seff	= fit.seff,
		bkg		= fit.bkg,
		L		= fit.L,
		higgs	= fit.higgs,
		beam	= fit.beam,
		isrcorr	= fit.isrcorr):
	pass
