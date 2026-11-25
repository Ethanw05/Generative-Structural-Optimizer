"""Isolated native mesher process. The generator retains the last valid part if the mesher fails."""
import sys
import numpy as np


def main():
    method,source,destination,value,envelope,threads=sys.argv[1:]
    with np.load(source) as mesh:
        vertices,triangles=mesh['v'],mesh['f'].astype(np.int32)
        holes=mesh['holes'] if 'holes' in mesh else []
    if method=='surface':
        import pytetwild
        diagonal=np.linalg.norm(np.ptp(vertices,axis=0))
        p,t=pytetwild.tetrahedralize(vertices,triangles,edge_length_abs=float(value),
            epsilon=float(envelope)/diagonal,stop_energy=14,num_opt_iter=12,
            num_threads=int(threads),vtk_ordering=True,quiet=True)
    elif method=='interior':
        import tetgen
        generator=tetgen.TetGen(vertices,triangles)
        for point in holes:generator.add_hole(point)
        # A cavity PLC can contain nearly coplanar, intentionally distinct facets.
        # Preserve their dividing edges while allowing boundary refinement on each facet.
        facet_flags='M' if len(holes) else ''
        p,t,*_=generator.tetrahedralize(switches=f'p{facet_flags}q1.6a{float(value):.12g}Q',steinerleft=-1)
    else:
        raise ValueError('Unknown volume verification method')
    np.savez_compressed(destination,p=p,tet=t)


if __name__=='__main__':main()
