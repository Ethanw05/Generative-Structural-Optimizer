"""Exact closest triangle and barycentric support on an indexed surface."""
import numpy as np
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from .cad_motion import polydata


class SurfaceLocator:
    def __init__(self,vertices,faces):
        self.vertices=vertices
        self.faces=faces
        self.locator=vtkStaticCellLocator()
        self.locator.SetDataSet(polydata(vertices,faces))
        self.locator.BuildLocator()

    def nearest(self,points):
        triangles=np.empty(len(points),dtype=np.intp)
        projections=np.empty_like(points,dtype=float)
        cell,sub,distance=reference(0),reference(0),reference(0.)
        hit=[0.,0.,0.]
        for i,point in enumerate(points):
            self.locator.FindClosestPoint(point,hit,cell,sub,distance)
            triangles[i]=int(cell)
            projections[i]=hit
        return triangles,projections
