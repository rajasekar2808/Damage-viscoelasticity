# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 22:19:37 2021

@author: gopalsamy
"""


#######################################################################
#########  EXAMPLE 2: 3 POINT BENDING ###########################

### VALIDATION OF GKV IMPLEMENTATION WITHOUT DAMAGE WITH ANALYTIC SOLUTION ##########
###  can VALIDATE FOR BOTH LINEAR AND NON-LINEAR CASE (non-linear -> asymmetric case)
### Analytical solution implemented in frequency domain (Correspondance principle) ...
### ... using (numerical) inverse Laplace transform (mpmath library using 'de Hoog et al.' method) from complex modulous of gkv model
#######################################################################

import os
import sys
sys.path.append('../lib')
#sys.path.append('./.')

import viscoelasticity_law2D as visclaw
import viscoelasticymech2d   as mech2d
from mesh import simplexMesh, dualLipMeshTriangle
import liplog
import logging

import mpmath as mpm
import numpy as np
import matplotlib.pylab as plt
import cProfile
import time as time_mod
import lipdamage as lipprojector


##############################################################################
#########################  DETAILs OF THE MESH ###############################
##############################################################################

############# load the mesh ##############
basemeshname = 'beam_slender';   ## name of the mesh
meshid = ''
meshfilename = '../msh/'+basemeshname + meshid +'.msh'
mesh_in_mm = True       ## True when mesh details are in 'mm' else False when in 'm'

########### Dimensions of the geometry ######################

## Dimensions
thickness = 20e-3; b= thickness;  ## thickness in z-direction 
L = 800/1000 
h = 40/1000



##############################################################################
##########################  FOR WRITING RESULTS TO FILE ######################
##############################################################################

### name for outputfiles 

simulation_id = str(1)     ## give new id for performing simulation on same mesh with different setting
expname = basemeshname+ simulation_id

### for outputfiles 
respath = '../tmp'
respath1 = os.path.join(respath,expname)
os.mkdir(respath1)  ## make sure a directory doesnt exist in the same name as expname


respath2 = os.path.join(respath1,'results')
os.mkdir(respath2)


###  initiate logger to log the actions
log_file_name = expname+'_log'
logger =  liplog.setLogger(respath = respath1, basefilename = log_file_name)



##################################################################
################# CONSTITUTIVE BEHAVIOUR #########################
##################################################################



#### Material properties  #######


## Material parameters



"""
ni = 1+1
E0=2674e6
E1=97e6
nu = 0.0
tau = 29
Yc = 30
 
E = [E0,E1]
tau =[tau]

"""



E = [31770, 87398, 123414, 65830, 62457, 62661, 7305, 12500, 418, 1743, 79, 39]    ## MPa
E = [i*1e6 for i in E]     ## Pa
tau = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 5e2, 1e3]   ## s
Yc = 23000    ## J/m^3
ni = 11+1     ## number of variables (excluding damage variable)  (or) number of units in GKV 

nu=0.0  ## Poisson's ratio

if len(tau)+1 != len(E) or len(E)!=ni:
    raise



##plane strain assumption
## find the material constants lamb and mu for each spring in GKV

lamb = []; mu=[]
for i in range(ni):
    lamb.append(E[i]*nu/(1.+nu)/(1.-2*nu))
    mu.append(E[i]/2./(1+nu))

lamb = tuple(lamb)
mu = tuple(mu)

damage_calc = False   ## set to true for performing damage calc.


## Load constitutive laws

unilateral = False  ## bool for asymmetric tension/compression effects
split_choice = None

if not unilateral:
    #### law 1: symmetric tension/compression  cons. law
    law = visclaw.viscoElasticity2dPlaneStrain(ni,lamb,mu,tau)
else:
    ### law 2: aymmetric tension/compression cons. law    
    if split_choice is None: split_choice = 3      ## determines the type of split
    law = visclaw.viscoElasticity2dPlaneStrain_ASSIM(ni,lamb,mu,tau,split_choice=split_choice)



##################################################
### Process the mesh and construct lip mesh  #####
##################################################


### start profiling

### start profiling
pr = cProfile.Profile()


logger.info('Starting Programm 3 point bending no damage')


mesh = simplexMesh.readGMSH(meshfilename)
if mesh_in_mm:  mesh.xy = mesh.xy/1000
nv = mesh.nvertices
nf =  mesh.ntriangles


logger.info('Mesh Loaded from file ' + meshfilename)
logger.info('Mesh size : nv: '+ str(nv) + ' nf: '+ str(nf))

### build lipMesh
lipmesh = dualLipMeshTriangle(mesh)
lipproj = lipprojector.damageProjector2D(lipmesh, verbose=True)
logger.info('LipMesh constructed from Mesh')
logger.info('LipMesh size : nv: '+ str(lipmesh.nvertices)+' nf: '+ str(lipmesh.ntriangles))

### build lipMesh
lipmesh = dualLipMeshTriangle(mesh)
lipproj = lipprojector.damageProjector2D(lipmesh, verbose=True)   ## projector for obtaining the bounds (upper  and lower bounds on damage)
logger.info('LipMesh constructed from Mesh')
logger.info('LipMesh size : nv: '+ str(lipmesh.nvertices)+' nf: '+ str(lipmesh.ntriangles))




##############################################################################
###################  INITIATE THE SOLVERS #####################################
#############################################################################

## create an instance of the mechanical class for viscoelasticity
mech = mech2d.Mechanics2D(mesh, law, lipproj,logger=logger)  


### DISPLACEMENT (AND INTERNAL STRAINS) SOLVER
solverdisp = None; solverdispoptions = None;
if not unilateral:
    ## load linear solver for displacements
     solverdisp = mech2d.Mechanics2D.solveDisplacementFixedDLinear
     solverdispoptions = {'linsolve':'cholmod'}
else:
    ## load non linear solver (works on Newton method with line search)
    solverdisp = mech2d.Mechanics2D.solve_u_eps_i_nonlinear
    solverdispoptions = {'linsolve':'cholmod','itmax':30, 'resmax':1.e-5,'res_energy_abs':1e-5}
    ## resmax - relative total force  on the free nodes;; 
    ## res_energy_abs- difference of incremental potential b/w 2 consective Newton iterations

### DAMAGE SOLVER
solverd = None; solverdoptions = None;
if damage_calc:
    solverd = mech2d.Mechanics2D.solveDLipBoundPatch 
    solverdoptions ={'mindeltad':1.e-3, 'fixpatchbound':False, 
                        'Patchsolver':'triangle', 'FMSolver':'triangle', 
                        'parallelpatch':False, 'snapthreshold':0.999,
                        #'kktsolveroptions': {'mode':'schur', 'linsolve':'cholmod'}
                        'kktsolveroptions': {'mode':'direct', 'linsolve':'umfpack'}
                        }


### Alternate Minimisation solver options (Staggered solver )
alternedsolver = mech.alternedSolver
alternedsolveroptions= {'abstole':1.e-8, 'reltole':1.e-3, 'deltadtol':1.e-2, 'max_iter': 20}


## Store all  options in a file to see later the options used 

opti_file = os.path.join(respath1, 'options.txt')


w2f = ['nb internal_variables : '+str(ni),'\nE : '+str(E) +' (Pa)', '\ntau: '+str(tau)+' (s)', '\nnu: '+str(nu), '\nYc: '+str(Yc),
       '\ndamage calc : '+str(damage_calc), '\nunilateral: '+str(unilateral), '\nsplit choice: '+str(split_choice),
       '\nmesh: '+meshfilename, '\nsolver disp options: '+str(solverdispoptions),'\ndamage solver options: '+str(solverdoptions),
       '\nalternate solver options : '+str(alternedsolveroptions) ]

with open(opti_file,'w') as fl:
    fl.writelines(w2f)




#############################################################################
################### BCD'S ###################################################
#############################################################################


idl0 = 12 #vid of left pin support
idl1 = 13 #vid of right pin support
idl2 = 14 ## physical id of loading line
## vertex id's of nodes
vidl0  = mesh.getClassifiedPoint(idl0)   
vidl1 = mesh.getClassifiedPoint(idl1)  
#vidl2  = mesh.getVerticesOnClassifiedEdges(idl2) #ids of the vertices on pulled line
vidl2 = mesh.getClassifiedPoint(idl2)  

if (len(vidl0)+len(vidl1)) ==0: logger.warning('No Dirichlet nodes')

imposed_displacement= dict()

# Dirichlet Boundary conditions. degree of freedom are indexed by vertice Id  : ux(id) -> u(id*2), uy(id)   = u(id*2+1)
for vid in vidl0:
    #imposed_displacement[2*int(vid)] = 0.
    imposed_displacement[2*int(vid)+1] = 0.

for vid in  vidl1:
    #imposed_displacement[2*int(vid)] =   0.     
    imposed_displacement[2*int(vid)+1] = 0.    

for vid in vidl2:
    imposed_displacement[2*int(vid)] = 0.
    imposed_displacement[2*int(vid)+1] = 0.
    


##############################################################
## function to calculate the reaction force on the supports
##############################################################
def calc_reaction_force(R):
    Fx_support = 0
    Fy_support = 0
    for vid0, vid1 in zip([vidl0],[vidl1]): 
        Fy_support += R[2*vid0+1]
        Fy_support += R[2*vid1+1]
        Fx_support += R[2*vid0]
        Fx_support += R[2*vid1]
    return (abs(Fx_support), abs(Fy_support))


def calc_reaction_force_from_stress(stress):
    ## calc reaction force from stress balance
    #print(mech.areas().shape, stress.shape)
    return np.dot(mech.areas() , stress)




### PARAMTERS FOR THE SIMULATION


velocity = [i/1000 for i in [ .5, 1, 10]]        ## m/s
max_displacement_y = [i/1000 for i in [5,5,10]]      ## max_displacement applied for each speed
time_step = [ .15, .1, .05] 
tot_time = []     ## total time when the max_displacement  is attained

"""
vi = int(2)
velocity = [velocity[vi]]
max_displacement_y = [max_displacement_y[vi]]
time_step = [time_step[vi]]
"""

w2f = ['\nvelocity : '+str(velocity), '\nTime step : '+ str(time_step)]
with open(opti_file,'a') as fl:
    fl.writelines(w2f)


### Analytical solution for  3 point bending beam

def beam_analytical_solution_PT(u_applied,speed,E,tau):
    ## analytical solution for ni =2 (PT model)
    g = (4*b*h**3/(L**3)) 
    
    
    F = g* (E[0]*E[1]/(E[0]+E[1]))*(u_applied  +(E[0]*tau[0]*speed/(E[0]+E[1]))*(1-np.exp((-u_applied/(speed*tau[0]))*((E[0]+E[1])/E[1]))) )
    return F


def dynam_mod_gkv(s):
    ## privdes dynamic modulous of GKV model ;; s =i*w ;; w the frequency
    E_s = 1/E[0]
    for i in range(len(tau)):
        E_s += 1/(E[i+1]*(1+tau[i]*s))
    return 1/E_s



def beam_analytical_solution(u_appl,speed, E, tau):
    ## analytical solution by inverse Laplace transform of dynamic modulous of GKV model
    ## inverse Laplace transform performed numerically using method = 'de Hoog et al.' from mpmath library
    g = (4*b*h**3/(L**3))
    
    load = lambda s: g*dynam_mod_gkv(s)*speed/s**2   ## load in frequency domain
    time_loc = u_appl/speed
    if time_loc[0] ==0.:
        time_loc[0] = 1e-12
    F = np.zeros_like(u_appl)
    for i in range(len(F)):
        F[i] = mpm.invertlaplace(load, time_loc[i])
    return F
    
    



## variables to store the results for force-displacement plot

uimp=[]
Fy  =[]



## plotting options
plt.close('all')
onlineplot = True      ## for saving contour plots to pdf
real_time_plot = True  ## for plotting  real time force-disp curves as the program runs
showmesh = False

if showmesh:
    mesh.plot()

if real_time_plot:
    plt.show()
    fig0, ax0 = plt.subplots(1,1)
    axes = plt.gca()
    axes.set_xlim(-1e-5, 10e-4)
    max_data_plot = 3
    axes.set_ylim(-2e-2, max_data_plot)
    line, = axes.plot(0,0)
    fig0.suptitle(r'$F(u)$')
    ax0.set_xlabel(r'Imposed Displacement (m)')
    ax0.set_ylabel(r'Reaction Force (N)')

## list to store time spent on disp and damage solver
timeu =[]
timed =[]

## list ot store number of iterations of alternate minimisation at each time step
alt_iter = []


## initital condition
u = np.zeros((nv,2))
d = np.zeros(nf)

##time increment 

for ind, (speed, max_uy, DT) in enumerate(zip(velocity, max_displacement_y, time_step)): 
    logger.info('\n\n\n\n Simulation '+ str(ind))
    logger.info("\n Message " +str(ind) +": Solving for velocity = "+ str(speed)+ " m/s ")
    R,eps_i = mech.zeros()
    time = 0
    count = 0
    temp_disp, temp_force = [0], [0] 
    timeu.append(0); timed.append(0);
    alt_iter_tmp = []
    ## reinitialize dirichlet conditions to zero for different speed setting
    for vid in  vidl2:
            imposed_displacement[2*int(vid) + 1] = 0
    u_appl= 0           # initial applied displacement
    while u_appl < max_uy:
        u_appl= speed*(time+DT)
        logger.info("\n\n\n Message " +str(ind) +": Solving for u = "+ str(u_appl)+ " m ")
        for vid in  vidl2:
            imposed_displacement[2*int(vid) + 1] =   -u_appl
        dmin = d.copy()
        pr.enable()
        res = alternedsolver( dmin =dmin, dguess =d.copy(), DT=DT, un= u.copy(), eps_i_n = eps_i.copy(), 
                    imposed_displacement = imposed_displacement, 
                    alternedsolveroptions = alternedsolveroptions,
                    solverdisp = solverdisp, solverdispoptions= solverdispoptions,
                    solverd= solverd, 
                    solverdoptions= solverdoptions,
                    damage_calc=damage_calc)
        if not res['Converged'] :
            print ('alterned Solver DID NOT converge at step ui =', speed*time, ' message is ', res['info'])
            logger.warning('alterned Solver DID NOT converge at step ui =', speed*time, ' message is ', res['info'])
            #u = res['u']
            #d = res['d']
            #mech.plots(u, d, eps1, u_appl, respath+'/'+expname+'LastBeforeFailure', showmesh = showmesh)
            #R = res['R']
            #break
        dtmp1 = res['d']
        dmind1 = np.linalg.norm(dtmp1-dmin)
        logger.info('\n End uimpminimize = '+ str(u_appl) + ' Conv :' + str(res['iter']) + ' iter, |dmin-d1|= ' + '%2.e'%dmind1+'\n\n')
        pr.disable()
        u = res['u']
        eps_i = res['eps_i']
        d = res['d']
        R = res['R']
        timeu[-1] += res['timeu']
        timed[-1] += res['timed']
        alt_iter_tmp.append(res['iter'])
        
        if onlineplot :     
            if np.max(d) > 0 :
                mech.plots(u, d,eps_i, u_appl, respath1+'/'+expname, showmesh = showmesh)
        


        ##updating local displacement curve
        temp_disp.append(u_appl)   
        temp_force.append(calc_reaction_force(R)[1]*thickness) 
        if real_time_plot:
            max_data_plot = max(max_data_plot, temp_force[-1])
            axes.set_ylim(-.2, 1.1*max_data_plot)
            axes.set_xlim(-1e-4, 1.1*temp_disp[-1])
            line.set_xdata(temp_disp)
            line.set_ydata(temp_force)       ## to scale the graph
            plt.draw()
            plt.pause(1e-10)   ## to update the plot
        
        if np.max(d) > .8:
            logger.warning("~0 N force attained. Hence simulation is stopped")
            break
        
        count+=1
        time+=DT
    
    
    tot_time.append(time)
    uimp.append(np.array(temp_disp))
    Fy.append(np.array(temp_force))  
    alt_iter.append(alt_iter_tmp)
    
    

     

plt.show()          

logger.info('Time spend on equilibrium solver :' + str(timeu) + 's')
logger.info('Time spend on damage solver      :' + str(timed) + 's')











def plot_analytical(speed,ax):
    marke_r = ['o', 'v', 's', '*', '>']
    for ind,i in enumerate(speed):
        t= np.linspace(0,tot_time[ind])
        u_applied = i*t
        F = beam_analytical_solution(u_applied,i ,E,tau)
        ax.plot(u_applied, np.array(F).squeeze(), marke_r[ind], label = 'analyt. '+ str(velocity[ind]*1000) + ' mm/s' )

def plot_force_displacement(force,disp,ax):
    
    for i,(f,u) in enumerate(zip(force,disp)):
        ax.plot(u,np.array(f).squeeze(), label = 'FEM '+ str(velocity[i]*1000) + ' mm/s')
        
def plot_f_d(ax):
    
    plot_force_displacement(Fy,uimp,ax)
    #plot_force_displacement_ref_data()
    plot_analytical(velocity,ax)
    ax.legend()



def plot_force_displacement_ref_data():
    ## refernce data from Benjamin's simulation of 2D beam bending
    ref_data = []
    for i in [1,2,4,5]:
        ref_data.append(np.load('../ref_data/data'+ str(i)+'.npy'))

    marke_r = ['o', 'v', 's', '*', '>']
    for i,data in enumerate(ref_data) :
        plt.plot(data[:,0]/1000,data[:,1], marke_r[i], label = 'ref '+ str(velocity[i]*1000) + ' mm/s')




def plot_vec(sol):
    mesh.plotVectorNodalField(sol[1].u, scale=.3)

#plt.figure(2)
fig,ax = plt.subplots()
plot_f_d(ax)
plt.xlabel('displacement (m)')
plt.ylabel('Force (N)')
fig.savefig(respath1+'/Force_displacement_.pdf', format = 'pdf') 
#plot_analytical(velocity)
#plt.legend()






