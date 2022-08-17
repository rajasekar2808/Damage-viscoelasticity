######### Step 1:  Fill the path to pvd files
########  Step 2: Open paraview and go to file-load state and load this python script
post_pr_file_name = 'output_post_proc.pvd'
path_to_file = r'D:\VBox shared folder\results_lf_7\post_proc\output_post_proc.pvd'









#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()



# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1097, 796]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [188.0, 50.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [188.0, 50.0, 751.6268509350543]
renderView1.CameraFocalPoint = [188.0, 50.0, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 194.53534383242547
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1097, 796)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
output_post_procpvd = PVDReader(registrationName= post_pr_file_name, FileName=path_to_file)
output_post_procpvd.CellArrays = ['damage', 'sx', 'sy', 'sxy']
output_post_procpvd.PointArrays = ['ux', 'uy']

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=output_post_procpvd)
calculator1.ResultArrayName = 'displacement vec'
calculator1.Function = ' ux*iHat+uy*jHat'

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=calculator1)
threshold1.Scalars = ['CELLS', 'damage']
threshold1.ThresholdRange = [0.0, 0.99]

# create a new 'Warp By Vector'
warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=threshold1)
warpByVector1.Vectors = ['POINTS', 'displacement vec']
warpByVector1.ScaleFactor = 2471.7862580582123

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from output_post_procpvd
output_post_procpvdDisplay = Show(output_post_procpvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
output_post_procpvdDisplay.Representation = 'Surface'
output_post_procpvdDisplay.ColorArrayName = ['POINTS', '']
output_post_procpvdDisplay.SelectTCoordArray = 'None'
output_post_procpvdDisplay.SelectNormalArray = 'None'
output_post_procpvdDisplay.SelectTangentArray = 'None'
output_post_procpvdDisplay.OSPRayScaleArray = 'ux'
output_post_procpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
output_post_procpvdDisplay.SelectOrientationVectors = 'None'
output_post_procpvdDisplay.ScaleFactor = 37.6
output_post_procpvdDisplay.SelectScaleArray = 'ux'
output_post_procpvdDisplay.GlyphType = 'Arrow'
output_post_procpvdDisplay.GlyphTableIndexArray = 'ux'
output_post_procpvdDisplay.GaussianRadius = 1.8800000000000001
output_post_procpvdDisplay.SetScaleArray = ['POINTS', 'ux']
output_post_procpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
output_post_procpvdDisplay.OpacityArray = ['POINTS', 'ux']
output_post_procpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
output_post_procpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
output_post_procpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
output_post_procpvdDisplay.ScalarOpacityUnitDistance = 17.410854520543804
output_post_procpvdDisplay.OpacityArrayName = ['POINTS', 'ux']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
output_post_procpvdDisplay.ScaleTransferFunction.Points = [-8.305818869587495e-08, 0.0, 0.5, 0.0, 6.835898897606532e-06, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
output_post_procpvdDisplay.OpacityTransferFunction.Points = [-8.305818869587495e-08, 0.0, 0.5, 0.0, 6.835898897606532e-06, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# restore active source
SetActiveSource(output_post_procpvd)
# ----------------------------------------------------------------

"""
if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='D:\VBox shared folder\extracts')
"""