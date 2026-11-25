"""Read named occurrences from the geometry-only importer and transform them into a design frame."""
import numpy as np
import trimesh
from .design_space import closed_solid


def occurrences(path):
    vertices=[];groups={};current=None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            fields=line.split()
            if not fields:continue
            if fields[0]=='o':
                current=' '.join(fields[1:])
                if current in groups:raise ValueError('Duplicate occurrence '+current)
                groups[current]=[]
            elif fields[0]=='v':
                vertices.append(tuple(map(float,fields[1:4])))
            elif fields[0]=='f':
                if current is None or len(fields)!=4:raise ValueError('Expected named triangulated geometry')
                groups[current].append([int(a.split('/')[0])-1 for a in fields[1:]])
    vertices=np.asarray(vertices)
    for name,tri in groups.items():
        used,ids=np.unique(tri,return_inverse=True)
        yield name,trimesh.Trimesh(vertices[used],ids.reshape(-1,3),process=True)


def obstacle_solids(scene,design_occurrence,transform,domain):
    result=[];bounds=np.array(domain.bounding_box()).reshape(2,3)
    found=False
    for name,mesh in occurrences(scene):
        if name==design_occurrence:
            found=True;continue
        mesh.apply_transform(transform)
        if np.any(mesh.bounds[0]>=bounds[1]) or np.any(mesh.bounds[1]<=bounds[0]):continue
        for body in mesh.split(only_watertight=False):
            if np.any(body.bounds[0]>=bounds[1]) or np.any(body.bounds[1]<=bounds[0]):continue
            result.append(closed_solid(body))
    if not found:raise ValueError('Selected design occurrence is missing from its assembly')
    return tuple(result)
