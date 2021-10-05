#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:14:27 2021

@author: rmojgani
"""
import numpy as np

def case_select(imperfect_type, case):
    lambdas = []
    lambdas_imperfect = []
    # KS
    if imperfect_type == 'KS':
        lambdas = [1.0,1.0,1.0]
        eqn_str = np.array( ['uu_{x}','u_{xx}','u_{xxxx}'] )
        if case == 1:
            lambdas_imperfect = [0.0,1.0,1.0]
        elif case == 2:
            lambdas_imperfect = [1.0,0.0,1.0]
        elif case == 3:
            lambdas_imperfect = [1.0,1.0,0.0]
        elif case == 4:
            lambdas_imperfect = [1.0,0.0,0.0]
        elif case == 5:
            lambdas_imperfect = [0.0,1.0,0.0]
        elif case == 6:
            lambdas_imperfect = [0.0,0.0,1.0]
        elif case == 7:
            lambdas_imperfect = [0.5,1.0,1.0]
        elif case == 8:
            lambdas_imperfect = [1.0,0.5,1.0]
        elif case == 9:
            lambdas_imperfect = [1.0,1.0,0.5]
        elif case == 10:
            lambdas_imperfect = [2.0,2.0,2.0]
        elif case == 11:
            lambdas_imperfect = [0.0,0.5,2.0] # might need revisiting
        elif case == 12:
            lambdas_imperfect = [0.5,0.0,1.0]     
            
    # Burgers
    elif imperfect_type == 'Burgers':
        lambdas = [1.0,-0.1,0.0]
        eqn_str = 'tobeassigned'
        lambdas_imperfect = [1.0,-0.1,0.0]
        
    elif imperfect_type == 'KSpu3x':
        lambdas = [1.0,1.0,1.0]
        eqn_str = np.array( ['uu_{x}','u_{xx}','u_{xxxx}','u_{xxx}'] )
        if case == 1:
            lambdas_imperfect = [1.0,1.0,1.0,0.1]
        elif case == 2:
            lambdas_imperfect = [1.0,1.0,1.0,0.5]
        elif case == 3:
            lambdas_imperfect = [1.0,1.0,1.0,1.0]
        elif case == 21:
            lambdas_imperfect = [0.0,1.0,1.0,0.5]
        elif case == 22:
            lambdas_imperfect = [1.0,0.0,1.0,0.5]
        elif case == 23:
            lambdas_imperfect = [1.0,1.0,0.0,0.5] 

  
    elif imperfect_type == 'KSpu3x_Du3':
        lambdas = [1.0,1.0,1.0]
        eqn_str = np.array( ['uu_{x}','u_{xx}','u_{xxxx}','u^2u_{x}'] )
        if case == 1:
            lambdas_imperfect = [1.0,1.0,1.0,1.5]
        elif case == 2:
            lambdas_imperfect = [1.0,1.0,1.0,0.15]
        elif case == 3:
            lambdas_imperfect = [1.0,1.0,1.0,0.015]
        elif case == 21:
            lambdas_imperfect = [0.0,1.0,1.0,0.15]
        elif case == 22:
            lambdas_imperfect = [1.0,0.0,1.0,0.15]
        elif case == 23:
            lambdas_imperfect = [1.0,1.0,0.0,0.15]
            
    elif imperfect_type == 'SH':
        # epsilon = 0.5
        # g = 1
        lambdas = [1.0,-1.0,1.0,2.0,1.0]
        eqn_str = np.array( ['u','u^2','u^3','u_{xx}','u_{xxxx}'] )
        if case == 1:
            lambdas_imperfect = [1.0,-1.0,1.0,2.0,1.0]


    elif imperfect_type == 'KS_p_u2uxx':
        lambdas = [1.0,1.0,1.0]
        eqn_str = np.array( ['uu_{x}','u_{xx}','u_{xxxx}','u^2u_{xx}'] )
        if case == 1:
            lambdas_imperfect = [1.0,1.0,1.0,1.5]
        elif case == 2:
            lambdas_imperfect = [1.0,1.0,1.0,0.15]
        elif case == 3:
            lambdas_imperfect = [1.0,1.0,1.0,0.015]
           
    return lambdas, lambdas_imperfect, eqn_str

