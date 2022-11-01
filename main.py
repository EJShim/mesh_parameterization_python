
import sys
import os
import vtk
import igl
from vtk.util import numpy_support
import numpy as np
import math
import random
from scipy.sparse import csr_matrix, coo_matrix


vtk_out = vtk.vtkOutputWindow()
vtk_out.SetInstance(vtk_out)


def tutte(V, F):

    #Get Boundary and Edge
    L = igl.boundary_loop(F)
    sizeL = len(L)


    #Iterate over boundary
    bc = []
    for i in range(sizeL):
        bc.append( [math.cos(math.pi*2/sizeL*i), math.sin( math.pi*2/sizeL*i )] )
    bc = np.array(bc, dtype=np.double)

    print("L: : ", L.shape)


    #Iterate over edge, build some kind of sparse matrix,, 
    E = igl.edges(F)
    
    I = []
    J = []
    Val = []

    diag = np.zeros( len(V) )
    for i, e in enumerate(E):
        tp = 1.0 / 3*  np.linalg.norm( (V[e[0]] - V[e[1]]) )

        I.append(e[0])
        J.append(e[1])
        Val.append(tp)

        I.append(e[1])
        J.append(e[0])
        Val.append(tp)

        diag[e[0]] -= tp
        diag[e[1]] -= tp
    
    #Add diag value to sparse matrix
    for i, v in enumerate(V):
        I.append(i)
        J.append(i)
        Val.append(diag[i])


    #A : Sparse matrix
    A = coo_matrix( (Val,(I,J)), shape=(len(V), len(V)) ).tocsr()
    B_flat = np.zeros((len(V),2), dtype=np.double)
    b = L.reshape(len(L), 1) #L -> Boundary
    Aeq = csr_matrix( (0, 0), dtype=np.double  )
    Beq = np.zeros(shape=(0,1))

    # print("A : ", A.shape)
    # print("B_flat : ", B_flat.shape)
    # print("b : ", b.shape)
    # print("bc : ", bc.shape)
    # print("Aeq: ", Aeq.shape)
    # print("Beq : ", Beq.shape)

    # print(A.shape)
    U = igl.min_quad_with_fixed(A, B_flat, b, bc, Aeq, Beq, False)

    return U[1], bc


def MakeActor(polydata):
    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputData(polydata)
    
    polydata.GetPointData().RemoveArray("Normals")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def MakeBoundaryActor(boundary):
    print(boundary.shape)
    boundary3D = np.concatenate( [boundary, np.zeros((len(boundary),1) )], axis=1)
    print(boundary3D)

    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData( numpy_support.numpy_to_vtk(boundary3D) )
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(np.arange(len(boundary))))
    
    mapper = vtk.vtkOpenGLSphereMapper()
    mapper.SetRadius(.005)
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(0, len(boundary))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


if __name__ == "__main__":
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    renWin.AddRenderer(ren)

    print("Helllo vtk and igl")

    if len(sys.argv) < 2:
        sys.argv.append("sample/sample.stl")


    #Read VTK file
    file_path = sys.argv[1]
    ext = file_path.split(".")[-1]

    if ext == "stl":
        reader = vtk.vtkSTLReader()
    elif ext == "obj":
        reader = vtk.vtkOBJReader()
    elif ext == "ply":
        reader = vtk.vtkPLYReader()
    else:
        raise(ext, "ext not supported")
    reader.SetFileName(sys.argv[1])
    reader.Update()
    polydata = reader.GetOutput()

    #Calculate curvature
    cc = vtk.vtkCurvatures()
    cc.SetInputData(polydata)
    cc.Update()
    polydata = cc.GetOutput()

    actor = MakeActor(polydata)
    actor.GetMapper().SetScalarRange(-1, 1)
    actor.SetScale(.02, .02, .02)
    ren.AddActor(actor)
    

    # Get V and F from polydatqa
    V = numpy_support.vtk_to_numpy( polydata.GetPoints().GetData())
    F = numpy_support.vtk_to_numpy( polydata.GetPolys().GetData() )
    F = F.reshape( int(len(F)/4), 4  )
    F = F[:, 1:]

    U_tutte, boundary = tutte(V, F)
    V_tutte = np.concatenate((U_tutte,  np.zeros( (len(U_tutte),1) ) ), axis=1) #Make three-dimensional

    boundaryActor = MakeBoundaryActor(boundary)
    ren.AddActor(boundaryActor)

    tutte_points = numpy_support.numpy_to_vtk( V_tutte )
    tuttePoly = vtk.vtkPolyData()
    tuttePoly.DeepCopy(polydata)
    tuttePoly.GetPoints().SetData(tutte_points)
    tutteActor = MakeActor(tuttePoly)
    tutteActor.GetMapper().SetScalarRange(-0.5, 0.5)
    # tutteActor.SetScale( 100, 100, 100 )

    ren.AddActor(tutteActor)

    renWin.Render()
    iren.Start()


