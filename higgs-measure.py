#! /sw/bin/pythonw
import string, math, random
import numpy as np
import matplotlib.pyplot as plt
import minuit as min
import subroutines as sub

x = sub.experiment(0.53,6.3,[125000,4.07,0.577],4200,125000*0.00004/math.sqrt(2.0),0.53)


