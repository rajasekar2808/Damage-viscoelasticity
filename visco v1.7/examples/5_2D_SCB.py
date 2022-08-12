# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 22:59:27 2022

@author: gopalsamy
"""



#######################################################################
#########  EXAMPLE 5: SCB 3 POINT BENDING (Mode -1 fracture)   ###########################

########## Can switch b/w  Lip-field and Phase-field AT2 damage solvers ########################
##########################################################################

## geometry from https://doi.org/10.1016/j.engfracmech.2022.108580

### ADAPTIVE TIME STEPPING USED TO ENSURE CONVERGENCE OF SOLUTION
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

import numpy as np
import matplotlib.pylab as plt
import cProfile
import time as time_mod
import lipdamage as lipprojector




##############################################################################
#########################  DETAILs OF THE MESH ###############################
##############################################################################

## load the mesh
basemeshname = 'scb_8'    ## name of the mesh
#basemeshname = 'scb - largeq'     ## coarse mesh (can use for the case without damage)
meshid = ''
meshfilename = '../msh/'+basemeshname + meshid +'.msh'
mesh_in_mm = True       ## True when mesh details are in 'mm' else False when in 'm'

########### Dimensions of the geometry ######################

## Dimensions
thickness = 50e-3; b= thickness;  ## thickness in z-direction 
R = 75e-3  ## radius
L = 2*R 
L_eff = 120e-3  ## lenght b/w supports
h = R   ##height




##############################################################################
##########################  FOR WRITING RESULTS TO FILE ######################
##############################################################################

### name for outputfiles 

simulation_id = str(1)     ## give new id for performing simulation on same mesh with different setting (to save in a different folder)
expname = basemeshname+ simulation_id

### for outputfiles 
#respath = '/data/OG2111300/lip_visco_results'
respath = '../tmp'
respath1 = os.path.join(respath,expname)
os.mkdir(respath1)  ## Note : make sure a directory doesnt exist in the same name as expname


respath2 = os.path.join(respath1,'results')
os.mkdir(respath2)


###  initiate logger to log the actions
log_file_name = expname+'_log'
mode = "a+"
# w: It is for write mode., r: It is for reading mode. a: It is for append mode.
# w+: Create the file if it does not exist and then open it in write mode.
# r+: Open the file in the read and write mode.
# a+: Create the file if it does not exist and open it in append mode
logger =  liplog.setLogger(respath = respath1, basefilename = log_file_name,mode= mode)


###############################################################################
######################## FOR RESUMING THE SIMULATION FROM OLDER RESULTS #####
################################################################################
resume_simulation = False
resume_path = None    ## provide resume path  for (internal) variables if resume_simulation is True
resume_path2 = None   ##  resume path for previosuly availble force-disp data 
resume_disp = None  ## provide previously applied displacement 


## example to resume simulation

"""
resume_simulation = True


resume_path = r'/data/OG2111300/lip_visco_results/scb_8_260722_1_2/results/results_u_0.0002000000000000002_.npz'
resume_path2 = r'/data/OG2111300/lip_visco_results/scb_8_260722_1_2/force_disp.npz'
resume_disp =  0.0002000000000000002   ## can extract from the file name of resume_path
"""
if resume_simulation:
    if resume_path is None or resume_path2 is None:
        raise('path to load files cant be empty')
    
    if resume_disp is None:
        raise('Provide the applied displacement for the loded result')
    
    u_appl = resume_disp 
    
    logger.info('\n\n Restarting the program at u='+str(u_appl)+ ' (m)')

    ld1 = np.load(resume_path)
    u = ld1['u']
    eps_i = ld1['eps_i']
    d = ld1['d']

    ld2 = np.load(resume_path2)
    uimp = [i for i in ld2['u'] if i<=resume_disp]
    Fx = [i for i in ld2['Fx']]
    Fy = [i for i in ld2['Fy']]
    alt_iter = [i for i in ld2['alt_iter']]  ## retreive nb of AM iterations
    energ = {'fe':list(ld2['fe']), 'vd': list(ld2['vd']), 'de':list(ld2['de']), 'wi':list(ld2['wi']), 'cl':list(ld2['cl'])}
    logger.info('Loaded results from the given path to resume simulation \n\n\n')



## stop the simulation when the post peak force reaces stop_force
stop_force = 50           ## N


##################################################################
################# CONSTITUTIVE BEHAVIOUR #########################
##################################################################



#### Material properties  #######


## Material parameters




ni = 4+1


E = [23569.55349514, 82001.07723347, 90114.27350565,  2978.80035906,
       55433.26590092, 27561.19715975, 37702.34687451,   499.72725469,
       13589.77741864,  6926.68471782]

E = [25395.19162914, 26640.4427727 ,  8183.85282941,   534.74609007, 41445.22071374] ##MPa


E = [ 9943.01599181,  1153.86800282,   130.42494908, 21131.71951743,
        9429.12744897]

E = [i*1e6 for i in E] ## Pa

tau = [1.10015863e-03, 2.52300991e-04, 3.57539478e+00, 1.00000000e-05,
       2.81680793e-02, 5.87776496e-03, 2.49202327e+01, 1.44960559e-01,
       6.86980921e-01]

tau = [1.46514373e-03, 6.01382312e-02, 6.42232624e+00, 1.00000000e-05] ## s

tau = [1.66927303e+01, 3.13276383e+02, 2.62989089e-02, 2.44254401e-01]

nu = 0.35



if len(tau)+1 != len(E) or len(E)!=ni:
    raise


Gc = 32.08  ## N/m
lc = 10e-3  ## characterstic/regularization length (m)

Yc = (3/4)*Gc/(2*lc)   ## for h(d) = 2d^2

##use lc = 2*lc1   ;;;  lc, lc1 - regularizing length for LF and PF simulations



##plane strain assumption
## find the material constants lamb and mu for each spring in GKV

lamb = []; mu=[];
for i in range(ni):
    lamb.append(E[i]*nu/(1.+nu)/(1.-2*nu))
    mu.append(E[i]/2./(1+nu))

lamb = tuple(lamb)
mu = tuple(mu)

damage_calc = 1   ## set to true for performing damage calc.


## Load constitutive laws

## (psi, phi) : (free energy, viscous dissipation) potentials
### law 1 :  f =  g1(d) psi               +    dt * g2(d)*phi + Y_c * h(d)    
### law 2 :  f =  g1(d) psi^+  +  psi^-   +    dt * g2(d)*phi + Y_c * h(d)

## law 1 - no asymmetric (tension/compression) effects   :::: law 2 - asymmetric effects


unilateral = True   ## bool for asymmetric tension/compression effects
split_choice = 2

if not unilateral:
    #### law 1: symmetric tension/compression  cons. law
    g = visclaw.GQuadratic()   ## degradation function g(d)
    H = visclaw.HQuadratic2()   ## softening function   h(d)
    law = visclaw.viscoElasticity2dPlaneStrain(ni,lamb,mu,tau,g1=g,g2=g,Gc = Gc,h=H,var_con=False)
else:
    ### law 2: aymmetric tension/compression cons. law    
    if split_choice is None: split_choice = 2      ## determines the type of split for free energy ( psi^+ and psi^-)
    g1 = visclaw.GQuadratic()    ## g1(d)
    g2 = visclaw.G_Const_1()     ## g2(d) = 1  (helps in numerical stability)
    H = visclaw.HQuadratic2()     ## h(d)
    law = visclaw.viscoElasticity2dPlaneStrain_ASSIM(ni,lamb,mu,tau,split_choice=split_choice,Gc = Gc,
                                                     g1 =g1,g2=g2,h=H)




##################################################
### Process the mesh and construct lip mesh  #####
##################################################


### start profiling
pr = cProfile.Profile()


logger.info('Starting Programm SCB 3 point bending')


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

damage_solver = 'LF'     ### = 'LF' for Lip-Field solver
#damage_solver = 'PF'     ### = 'PF' for Phase-field AT2 solver


## create an instance of the mechanical class for viscoelasticity
mech = mech2d.Mechanics2D(mesh, law, lipproj,logger=logger,lc = lc)  


### DISPLACEMENT (AND INTERNAL STRAINS) SOLVER
solverdisp = None; solverdispoptions = None;
if not unilateral:
    ## load linear solver for displacements
     solverdisp = mech2d.Mechanics2D.solveDisplacementFixedDLinear
     solverdispoptions = {'linsolve':'cholmod'}
else:
    ## load non linear solver (works on Newton method with line search)
    solverdisp = mech2d.Mechanics2D.solve_u_eps_i_nonlinear
    solverdispoptions = {'linsolve':'cholmod','itmax':20, 'resmax':1.e-12,'res_energy_abs':1e-8}
    ## resmax - relative total force  on the free nodes;; 
    ## res_energy_abs- difference of incremental potential b/w 2 consective Newton iterations

### DAMAGE SOLVER
solverd = None; solverdoptions = None;
if damage_calc:
    if damage_solver == 'PF':
        solverd = mech2d.Mechanics2D.phase_field_AT2
        solverdoptions ={'linsolve':'cholmod'}
    else:
        solverd = mech2d.Mechanics2D.solveDLipBoundPatch 
        solverdoptions ={'mindeltad':1.e-3, 'fixpatchbound':False, 
                            'Patchsolver':'triangle', 'FMSolver':'triangle', 
                            'parallelpatch':False, 'snapthreshold':0.999,
                            #'kktsolveroptions': {'mode':'schur', 'linsolve':'cholmod'}
                            'kktsolveroptions': {'mode':'direct', 'linsolve':'umfpack'}
                            }


### Alternate Minimisation solver options (Staggered solver )
alternedsolver = mech.alternedSolver
alternedsolveroptions= {'abstole':1.e-8, 'reltole':1.e-3, 'deltadtol':1.e-3, 'max_iter': 50}





#############################################################################
################### BCD'S ###################################################
#############################################################################




## BCD's for displacement

## vertex id's for BCD's
##  idl0 - physical tag of left pin support line
##  idl1 - physical tag of right roller support line
##  idl2 - physical tag of applied displacement at top (loading  line)
idl0 = 101
idl1 = 102
idl2 = 103

## node number of physical tags / vertex ids
vidl0 = mesh.getVerticesOnClassifiedEdges(idl0)
vidl1 = mesh.getVerticesOnClassifiedEdges(idl1)
vidl2 = mesh.getVerticesOnClassifiedEdges(idl2)


"""
velocity = [i/1000 for i in [.1,.001,.5, 1, 10]]        ## m/s
max_displacement_y = [i/1000 for i in [200,200,200,200,10]]      ## max_displacement applied for each speed
time_step = [.05,.5 ,.001, .01, .05] 
"""

speed = 1/1000/60   ##   mm/min -> m/s
max_uy = 100/1000   ## maximum allowed displacement for the simulation 
DT = .2    ##  time step (s) 
adap_time_step = False        ### set to True for adaptive time stepping to ensure proper cpnvergence (might take longer time)




#############################################################################
## Store all  options in a text file to remember the options used ###############
#############################################################################

opti_file = os.path.join(respath1, 'options.txt')


w2f = ['nb internal_variables : '+str(ni),'\nE : '+str(E) +' (Pa)', '\ntau: '+str(tau)+' (s)', '\nnu: '+str(nu), '\nGc: '+str(Gc),
       '\nlc : '+str(lc)+' (m)','\nrestart : '+str(resume_simulation),'\nrestart_path : '+str(resume_path),'\nresume_disp : '+str(resume_disp)+' (m)' ,
       '\ndamage calc : '+str(damage_calc), '\nunilateral: '+str(unilateral), '\nsplit choice: '+str(split_choice),
       '\nmesh: '+meshfilename, '\nsolver disp options: '+str(solverdispoptions),'\ndamage solver options: '+str(solverdoptions),
       '\nalternate solver options : '+str(alternedsolveroptions) ]

with open(opti_file,'w') as fl:
    fl.writelines(w2f)


w2f = ['\nvelocity : '+str(speed), '\nTime step : '+ str(DT),'\nAdaptive time stepping : '+str(adap_time_step)]
with open(opti_file,'a') as fl:
    fl.writelines(w2f)





        
#############################################################################
#####  ELEMENTS WHERE DAMAGE IS FORCE TO ZERO TO AVOID BOUNDARY EFFECTS #####
#############################################################################

## elements where damage is enforced zero 
## (to prevent damage close to supports or to save computation time if crack path is known in advance )

videl_d0  =None

videl_d0 = []
a_low = -np.inf/1000     ;   a_left = -np.inf/1000
a_uppe = np.inf/1000     ;   a_right = np.inf/1000

## damage is possible only when x, y in the  domain (of the geometry) satisfies following
##    a_low < y < a_upper   and  a_left < x < a_right    (else d = 0.)

for ele_num,ele in enumerate(mesh.triangles):
    for ver in ele:
        x_ver = mesh.xy[ver,0]
        y_ver = mesh.xy[ver,1]
        if y_ver<a_low or y_ver> a_uppe or x_ver< a_left or x_ver > a_right :
            videl_d0.append(ele_num)
            break





if (len(vidl0)+len(vidl1)) ==0: 
    logger.warning('No Dirichlet nodes')
    raise("Please provide BCD's")


imposed_displacement= dict()

# Dirichlet Boundary conditions. degree of freedom are indexed by vertice Id  : ux(id) -> u(id*2), uy(id)   = u(id*2+1)
for vid in vidl0:
    imposed_displacement[2*int(vid)] = 0.
    imposed_displacement[2*int(vid)+1] = 0.

for vid in  vidl1:
    #imposed_displacement[2*int(vid)] =   0.     
    imposed_displacement[2*int(vid)+1] = 0.    

for vid in vidl2:
    #imposed_displacement[2*int(vid)] = 0.
    imposed_displacement[2*int(vid)+1] = 0.
    

##############################################################
## function to calculate the reaction force on the supports
##############################################################
def calc_reaction_force(R):
    Fx_support = 0
    Fy_support = 0
    for vid0, vid1 in zip(vidl0,vidl1): 
        Fy_support += R[2*vid0+1]
        Fy_support += R[2*vid1+1]
        Fx_support += R[2*vid0]
        Fx_support += R[2*vid1]
    return (abs(Fx_support), abs(Fy_support))


def calc_reaction_force_from_stress(stress):
    ## calc reaction force from stress 
    #print(mech.areas().shape, stress.shape)
    return np.dot(mech.areas() , stress)







## plotting options
plt.close('all')
onlineplot = False      ## for saving contour plots to pdf (takes too much space !!)
real_time_plot = False  ## for plotting  real time force-disp curves as the program runs
showmesh = False

if showmesh:
    mesh.plot()

if real_time_plot:
    plt.show()
    fig0, ax0 = plt.subplots(1,1)
    axes = plt.gca()
    axes.set_xlim(-1e-5, 20)
    max_data_plot = 200
    axes.set_ylim(-1e-2, max_data_plot)
    line, = axes.plot(0,0)
    fig0.suptitle(r'$F(u)$')
    ax0.set_xlabel(r'Imposed Displacement (mm)')
    ax0.set_ylabel(r'Reaction Force (kN)')

## list to store time spent on disp and damage solver
timeu =[]
timed =[]



##time increment 

logger.info("\n Message : Solving for velocity = "+ str(speed)+ " m/s ")
if resume_path is None:
    R,eps_i = mech.zeros()
    uimp = [0]
    Fx = [0]
    Fy =[0]
    alt_iter = [0]
    u_appl= 0 
    u = np.zeros((nv,2))
    d = np.zeros(nf)
    ## dict vaariable to store energy of bulk at a given time step
    energ = {'fe':[0], 'vd': [0], 'de':[0], 'wi':[0], 'cl': [0]}    ## J
else:
    R = np.zeros((nv,2))
time = 0
count = 0
 
timeu.append(0); timed.append(0);




## adaptive time stepping variable
DT1 = DT; 
while u_appl < max_uy:
    u_appl += speed*(DT1)
    
    logger.info("\n\n\n Message : Solving for u = "+ str(u_appl)+ " m " + " (time step index = "+str(count)+")")
    for vid in  vidl2:
        imposed_displacement[2*int(vid) + 1] =   -u_appl
    dmin = d.copy()
    pr.enable()
    res= alternedsolver( dmin =dmin, dguess =d.copy(), DT=DT1, un= u.copy(), eps_i_n = eps_i.copy(), 
                imposed_displacement = imposed_displacement, 
                alternedsolveroptions = alternedsolveroptions,
                solverdisp = solverdisp, solverdispoptions= solverdispoptions,
                solverd= solverd, 
                solverdoptions= solverdoptions,
                damage_calc=damage_calc, imposed_d_0=videl_d0,adap_time_step=adap_time_step)
    
    
       
        
    if res['ad_tim']:
        ad_tim_count = (np.log(DT/DT1)/np.log(2))
        if ad_tim_count > 100:  # max count for ad_time_stepping
            logger.error('\n Failed with adaptive time stepping. Reached maximum allowed '+
                         'count for adaptive time stepping!')
            
            break
        logger.info('\t Starting adaptive time stepping :'+str(ad_tim_count)+' \n')
        u_appl -= speed*DT1   ## revert back u_appl to solve for new u_appl
        DT1 = DT1/2
        
    else:
        
        if not res['Converged'] :
            #print ('alterned Solver DID NOT converge at step ui =', speed*time, ' message is ', res['info'])
            logger.warning('alterned Solver DID NOT converge at step ui =', speed*time, ' message is ', res['info'])
        dtmp1 = res['d']
        dmind1 = np.linalg.norm(dtmp1-dmin)
        logger.info('\n End uimpminimize = '+ str(u_appl) + ' Conv :' + str(res['iter']) + ' iter, |dmin-d1|= ' + '%2.e'%dmind1+'\n\n\n')
        pr.disable()
        u = res['u']
        eps_i_n = eps_i.copy()
        eps_i = res['eps_i']
        d = res['d']
        R = res['R']
        d_nodal = res['d_nodal']
        alt_iter.append(res['iter'])  ## number of AM iterations
        stress = law.trialStress(mech.strain(u), eps_i,d)
        timeu[-1] += res['timeu']
        timed[-1] += res['timed']
        
        if onlineplot :     
            if np.max(d) > 0 and count%4==0:
                mech.plots(u, d,eps_i, u_appl, respath2+'/'+expname, showmesh = showmesh,DT=DT1,eps_i_n=eps_i_n)
        
        file1 =respath2+'/'+'results_u_'+str(u_appl)+'_.npz'
        logger.info("Saving output to file")
        np.savez(file1,u=np.array(u).squeeze(),d=np.array(d).squeeze(),DT = np.array([DT1]),
                     eps_i = np.array(eps_i).squeeze(),R = res['R'], stress = stress)
        ##updating local displacement curve
        uimp.append(u_appl)
        Fx.append(calc_reaction_force(R)[0]*thickness)
        Fy.append(calc_reaction_force(R)[1]*thickness)
        ene = mech.energies(u, eps_i, eps_i_n,d,DT1,d_nodal)
        energ['fe'].append(ene['fe'] *thickness)
        energ['vd'].append(energ['vd'][-1]+ ene['vd'] *thickness)
        energ['de'].append(ene['de'] *thickness)
        energ['wi'].append(energ['wi'][-1]+ Fy[-1]*speed*(DT1))
        energ['cl'].append(mech.crack_length(d,d_nodal))
        if real_time_plot:
            axes.set_xlim(-1e-4, max(uimp))
            axes.set_ylim(-2, max(Fy))
            line.set_xdata(uimp)
            line.set_ydata(Fy)
            plt.draw()
            plt.pause(1e-14)  ## to update the plot
        
        ## save force-displacement and nb. of AM iterations and update it at each time step
        np.savez(respath1+'/force_disp.npz',Fx = np.array(Fx).squeeze(), Fy = np.array(Fy).squeeze(),
                 u = np.array(uimp).squeeze(), alt_iter = np.array(alt_iter).squeeze(), 
                 fe = np.array(energ['fe']), vd = np.array(energ['vd']),de = np.array(energ['de']),
                 wi = np.array(energ['wi']),cl = np.array(energ['cl']))
        
        
        count+=1
        time+=DT1
        ## revert back time step to initial time step for the next time iteration once the results converged
        DT1 = DT
        
        if Fy[-1] < stop_force and count > 150: 
            logger.warning("~0 N ("+str(Fy[-1]) + " N) force attained. Hence simulation is stopped")
            break
        

        
        
        
        
#ax0.plot(np.array(temp_disp), np.array(temp_force), '-g')

logger.info('Time spend on equilibrium solver :' + str(timeu) + 's')
logger.info('Time spend on damage solver      :' + str(timed) + 's')



### to load back the files
"""
ld = np.load(file1)
cra_length = ld['cl']
"""




def plot_f_d(ax):
    
    plot_force_displacement2(Fy,uimp,ax)
    ax.legend()

def plot_force_displacement2(force,disp,ax):
    ax.plot(disp,force, label = 'FEM '+ str(speed*1000*60) + ' mm/min')


#plt.figure(2)

fig,ax = plt.subplots()
plot_f_d(ax)
plt.xlabel('displacement (m)')
plt.ylabel('Force (N)')
fig.savefig(respath1+'/Force_displacement_.pdf', format = 'pdf') 
plt.legend()