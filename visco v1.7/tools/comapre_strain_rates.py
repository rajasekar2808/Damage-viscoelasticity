# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:30:52 2021

@author: gopalsamy
"""

### Post processing to compare results for TDCB specimen 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
"""


font = {'size'   : 16}

matplotlib.rc('font', **font)

respath = './tmp_T/'
#respath = './tmp/'

test = ['tdcb_r1_lc_0.005_speed_1e-05','tdcb_r1_lc_0.005_speed_0.0001','tdcb_r1_lc_0.005_speed_0.001']


#test = ['tdcb_r1_lc_0.005_speed_0.0001_Temp_10','tdcb_r1_lc_0.005_speed_0.0001','tdcb_r1_lc_0.005_speed_0.0001_Temp_30_new_1']

speed = [1e-5, 1e-4, 1e-3, 1e-2]

temperature = [10,20,30]

dt = [1e-2, 1e-3, 1e-4, 1e-5]


def plotglobal(fig, ax1, uimp,Fy,cracklength, step = -1,speed=0):
    colora = 'tab:red'
    colorf =  'tab:blue'
    ax1.set_xlabel('Imposed displacement (mm)')
    ax1.set_ylabel('Load (kN)', color=colorf)  # we already handled the x-label with ax1
    #ax1.set_xlim(0., gu.max()) 
    #ax1.set_ylim([-10e-3,6])
    #ax1.set_xlim([-1e-7, .0003])
    ax1.plot(uimp, Fy,label='Force', color=colorf)
    ax1.tick_params(axis='y', labelcolor=colorf)
    #ax1.legend(title = '\n')
    if step >= 0 :
        ax1.plot(uimp[step], Fy[step],'o',  color=colorf)
         
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(uimp, cracklength, label = 'crack length',color=colora)
    ax2.tick_params(axis='y', labelcolor=colora)
    ax2.set_ylabel('Crack length (mm)', color=colora)
    if step >= 0 :
        ax2.plot(uimp[step], cracklength[step], 'o', color=colora)
        #ax2.legend(title = '\n')
    ax1.set_title(' displacement rate =' + str(speed)+ ' mm/s')
    handles, labels = ax1.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center')
    handles, labels = ax2.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower left')
    #ax1.set_xlim(0,2.)
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    
    return fig, ax1


def force_u_crack(time_ind = None):
    ##superimose F vs disp and cl vs disp
    
    ind = 2
    test2 = [test[ind]]
    fig,ax1 = plt.subplots()
    plt.show()
    figManager= plt.get_current_fig_manager()
    figManager.window.showMaximized()   
    #fig.tight_layout()
    #plt.switch_backend('QT5')
    ax = (ax1,)
    if time_ind is None:
        time_ind= 1500
    for i,fname in enumerate(test2):
        f1 = np.load(respath+fname+'/'+fname+'_RU_.npz')
        #f2 = respath+fname+'/'+fname+'_energies_.npy'
        u = f1['u']*1000    
        Fy = f1['Fy']/1000
        cl = f1['cl']*1000
        plotglobal(fig, ax[i], u, Fy, cl, time_ind, speed[ind]*1000)
    #fig.tight_layout()
    
        


def force_disp_superimposed():
    fig,ax = plt.subplots()
    ax.set_title('Force vs displacement')
    ax.set_xlabel('imposed displacement (mm)')
    ax.set_ylabel('Force (kN)')
    name = './post_proc/'
    
    for i,fname in enumerate(test):
        f1 = np.load(respath+fname+'/'+fname+'_RU_.npz')
        #f2 = respath+fname+'/'+fname+'_energies_.npy'
        u = f1['u']*1000    
        Fy = f1['Fy']/1000
        #ax.plot(u,Fy,label= speed[i]*1000)
        ax.plot(u,Fy,label= temperature[i])
        
        
    
    #ax.legend(title = 'loading rate (mm/s)')
    ax.legend(title = 'Temperature (' + r"$^o$" +'C)')
    fig.savefig(name+'force_displ_compar_.pdf', format = 'pdf')
    #fig.savefig(name+'force_displ_compar_Temperature.pdf', format = 'pdf')
    plt.show()


def crack_length_disp_superimposed():
    fig,ax = plt.subplots()
    ax.set_title('crack length vs imposed displacement', fontsize = 18)
    #ax.set_title('loading rate = .1 mm/s', fontsize = 18)
    ax.set_xlabel('imposed displacement (mm)', fontsize = 18 )
    ax.set_ylabel('crack length (mm)', fontsize = 18)
    name = './post_proc/'
    
    for i,fname in enumerate(test):
        f1 = np.load(respath+fname+'/'+fname+'_RU_.npz')
        #f2 = respath+fname+'/'+fname+'_energies_.npy'
        u = f1['u']*1000    
        cl = f1['cl']*1000
        #ax.plot(u,cl,label= speed[i]*1000)
        ax.plot(u,cl,label= temperature[i])
        
        
    
    #ax.legend(title = 'loading rate (mm/s)')
    ax.legend(title = 'Temperature (' + r"$^o$" +'C)', fontsize = 18)
    fig.savefig(name+'cl_displ_compar_.pdf', format = 'pdf')
    #fig.savefig(name+'cl_displ_compar_Temperature.pdf', format = 'pdf')
    plt.show()

def plot_energies(work_input, free_energy, visc_dissip, damage_dissip,DT,fig = None, ax = None ):
    if ax is None:
        fig,ax = plt.subplots()
    ## plot of different energies (/m2 cross section of homogenous bar)
    del_wi = np.array(work_input)
    del_fe = np.array([0]+[free_energy[i+1]-free_energy[i] for i in range(len(free_energy)-1)])
    del_visc = np.array(visc_dissip)
    del_de = np.array([0]+[damage_dissip[i+1] - damage_dissip[i] for i in range(len(damage_dissip)-1)])
    
    ax.plot(del_wi)
    ax.plot(del_fe)
    ax.plot(del_visc)
    ax.plot(del_de)
    
    ax.set_xlabel('time step indices with DT = '+ str(DT)+ ' s')
    ax.set_ylabel('Incremental energies (J)')
    
    ax.plot(del_fe+del_visc+del_de, '*')
    ax.set_title('Incremental energies (Nm)')
    
    ax.legend(['Work input (WU)','free energy (fe)','viscous dissip. (vd)','damage dissip. (dd)','fe +vd + dd'])
    return fig, ax    
    

def disp_crack_velocity_superimposed():
    tot_time = [3.9699999999999593, 0.5280000000000004, 0.060800000000000715,0.002980000000000007]
    fig,ax = plt.subplots()
    ax.set_title('crack velocity vs displacement')
    ax.set_xlabel('displacement (m)')
    ax.set_ylabel('crack velocity (m/s)')
    name = './post_proc/'
    
    #symb = ['*', 'o', '>']
    
    for i,fname in enumerate(test):
        f1 = np.load(respath+fname+'/'+fname+'_RU_.npz')
        #f2 = respath+fname+'/'+fname+'_energies_.npy'
        u = f1['u']    
        cl = f1['cl']
        
        cv = [0]+list(np.diff(np.array(cl))/dt[i])
        
        ax.plot(u,cv,label= speed[i])
        
        
    
    ax.legend(title = 'loading rate (m/s)')
    fig.savefig(name+'crack_velocity_displacement_.pdf', format = 'pdf')
    plt.show()

    


#force_disp_superimposed()
#force_u_crack()

#disp_crack_velocity_superimposed()