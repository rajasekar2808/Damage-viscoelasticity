Simulation of damage softening in viscoelasticity (GKV model) using Lip-field and phase field (AT2) regularization

- Example scripts to run the code can be found in examples folder
- Constitutive law file, mechanical file, mesh file, lip and phase damage files in the lib folder
- tools folder contain files for post process (to open the results in paraview) and files for least square fitting of GKV parameters from Dynamics modulous (or Prony series parameters)
- meshes used for the simulations could be found in the mesh folder. Mesh generated using GMsh (https://gmsh.info/)
- tmp folder to store the output results 
- All examples are run for SI units 


Installation:

1)  Install Python v3  ( https://www.python.org/downloads/windows/     or    https://www.spyder-ide.org/)
2)  Install package installer for python by executing   'get-pip.py'    (command line:   python3 get-pip.py ;   pip install --upgrade pip;   )
3)  Installation of nesseary libraries uisng pip install   
	(  command line:   pip install cvxopt; pip install mpmath; pip install scipy; pip install sympy;  pip install numpy; pip install matplotlib, pip install pylab; pip install triangle; pip install logging; pip install pyevtk; )