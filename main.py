
import sys
import vtk
import igl
from vtk.util import numpy_support
import numpy as np
import math
from scipy.sparse import csr_matrix, coo_matrix


def tutte(V, F):

    #Get Boundary and Edge
    L = igl.boundary_loop(F)
    E = igl.edges(F)
    sizeL = len(L)


    #Iterate over boundary
    bc = []
    for i in range(sizeL):
        bc.append( [math.cos(math.pi*2/sizeL*i), math.sin( math.pi*2/sizeL*i )] )
    bc = np.array(bc, dtype=np.double)


    #Iterate over edge, build some kind of sparse matrix,, 
    I = []
    J = []
    Val = []

    diag = np.zeros( len(V) )
    for i, e in enumerate(E):
        tp = 1.0 / np.linalg.norm( (V[e[0]] - V[e[1]]) )

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

    return U[1]


def MakeActor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)


    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor



if __name__ == "__main__":
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(512, 512)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    print("Helllo vtk and igl")

    if len(sys.argv) < 2:
        sys.argv.append("//192.168.0.113/Imagoworks/Data/confident/Mesh/IntraoralScan/DAEYOU-cut/train/2930/136.vtp")


    #Read VTK file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(sys.argv[1])
    reader.Update()
    polydata = reader.GetOutput()
    actor = MakeActor(polydata)
    ren.AddActor(actor)
    # polydata.GetPointData().RemoveArray("Normals")


    # Get V and F from polydatqa
    V = numpy_support.vtk_to_numpy( polydata.GetPoints().GetData())
    F = numpy_support.vtk_to_numpy( polydata.GetPolys().GetData() )
    F = F.reshape( int(len(F)/4), 4  )
    F = F[:, 1:]

    U_tutte = tutte(V, F)
    V_tutte = np.concatenate((U_tutte,  np.zeros( (len(U_tutte),1) ) ), axis=1) #Make three-dimensional


    tutte_points = numpy_support.numpy_to_vtk( V_tutte )
    tuttePoly = vtk.vtkPolyData()
    tuttePoly.DeepCopy(polydata)
    tuttePoly.GetPoints().SetData(tutte_points)
    tutteActor = MakeActor(tuttePoly)
    tutteActor.SetScale( 100, 100, 100 )

    ren.AddActor(tutteActor)

    renWin.Render()
    iren.Start()


