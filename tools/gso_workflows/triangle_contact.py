"""Exact narrow-phase triangle contact classification for a floating-point broad phase."""
from fractions import Fraction


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def interval(triangle,distances,direction):
    projection=[dot(v,direction) for v in triangle]
    points=[q for q,d in zip(projection,distances) if d==0]
    for i,j in ((0,1),(1,2),(2,0)):
        a,b=distances[i],distances[j]
        if a*b<0:
            points.append(Fraction(a*projection[j]-b*projection[i],a-b))
    if points:
        return min(points),max(points)
    return None


def coplanar_overlap(a,b,shared,normal):
    axis=max(range(3),key=lambda i:abs(normal[i]))
    def project(v):return tuple(x for i,x in enumerate(v) if i!=axis)
    def turn(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    polygon=list(map(project,a))
    clip=list(map(project,b))
    if turn(*clip)<0:clip.reverse()
    for p,q in zip(clip,clip[1:]+clip[:1]):
        out=[]
        for x,y in zip(polygon,polygon[1:]+polygon[:1]):
            dx,dy=turn(p,q,x),turn(p,q,y)
            if dx>=0:out.append(x)
            if dx*dy<0:
                t=Fraction(dx,dx-dy)
                out.append(tuple(xi+t*(yi-xi) for xi,yi in zip(x,y)))
        polygon=out
        if not polygon:return False
    common=list(map(project,shared))
    if not common:return True
    if len(common)==1:return any(p!=common[0] for p in polygon)
    if len(common)>2:return True
    start,end=common
    delta=sub(end,start)
    return any(turn(start,end,p)!=0 or not 0<=dot(sub(p,start),delta)<=dot(delta,delta) for p in polygon)


def improper_contact(vertices,face_a,face_b):
    indices=list(dict.fromkeys([*map(int,face_a),*map(int,face_b)]))
    ratios={i:[float(x).as_integer_ratio() for x in vertices[i]] for i in indices}
    denominator=max(q for v in ratios.values() for _,q in v)
    points={i:tuple(p*(denominator//q) for p,q in row) for i,row in ratios.items()}
    a=[points[int(i)] for i in face_a]
    b=[points[int(i)] for i in face_b]
    na=cross(sub(a[1],a[0]),sub(a[2],a[0]))
    nb=cross(sub(b[1],b[0]),sub(b[2],b[0]))
    if not any(na) or not any(nb):return True
    da=[dot(nb,sub(v,b[0])) for v in a]
    db=[dot(na,sub(v,a[0])) for v in b]
    if min(da)>0 or max(da)<0 or min(db)>0 or max(db)<0:return False
    shared=[points[int(i)] for i in set(face_a).intersection(face_b)]
    direction=cross(na,nb)
    if not any(direction):
        return coplanar_overlap(a,b,shared,na)
    ia,ib=interval(a,da,direction),interval(b,db,direction)
    if not ia or not ib:return False
    lo,hi=max(ia[0],ib[0]),min(ia[1],ib[1])
    if hi<lo:return False
    permitted=[dot(p,direction) for p in shared]
    return not permitted or lo<min(permitted) or hi>max(permitted)
