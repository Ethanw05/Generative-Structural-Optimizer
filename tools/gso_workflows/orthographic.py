"""Orthographic CPU renderer for the actual boundary triangles and elemental stresses."""
import ctypes
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def rasterizer(executable):
    name = {'win32':'gso_view.dll', 'darwin':'libgso_view.dylib'}.get(sys.platform,'libgso_view.so')
    library = ctypes.CDLL(str(Path(executable).resolve().with_name(name)))
    array = np.ctypeslib.ndpointer
    function = library.gsoRasterize
    function.argtypes = [ctypes.c_size_t, array(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_size_t, array(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),
                         array(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_int,
                         array(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS')]
    function.restype = ctypes.c_int
    return function


def color_positions(values,maximum,log_reference=None):
    values=np.clip(values,0.,maximum)
    if maximum<=0. or not np.isfinite(maximum):
        raise ValueError('Invalid stress color scale')
    if log_reference is None:return values/maximum
    if log_reference<=0. or not np.isfinite(log_reference):
        raise ValueError('Invalid logarithmic stress reference')
    return np.log1p(values/log_reference)/np.log1p(maximum/log_reference)


def render_views(cfg, inputs, executable):
    rasterize = rasterizer(executable)
    views = list(zip(cfg['target'],cfg['back'],cfg['right'],cfg['width'],strict=True))
    linear = cfg['palette'][:,:3]
    palette = np.where(linear <= .0031308,12.92*linear,1.055*linear**(1./2.4)-.055)
    width, height = 2000, 720
    for source,prefix in inputs:
        with np.load(source) as data:
            v = data['v']
            f = np.ascontiguousarray(data['f'],dtype=np.int32)
            if 'vm' in data:
                log_reference=float(cfg['log_reference']) if 'log_reference' in cfg else None
                bins=np.rint(255*color_positions(data['vm'],float(cfg['scale']),log_reference)).astype(int)
                color = np.rint(255*palette[bins]).astype(np.uint8)
                normals = None
            else:
                tri = v[f]
                normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
                normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None],np.finfo(float).tiny)
                del tri
        for index,(target,back,right,span) in enumerate(views):
            up = np.cross(back,right)
            view = (v-target) @ np.stack((right,-up,back)).T
            view[:,:2] = view[:,:2]*(width/span)+[width/2.,height/2.]
            if normals is not None:
                light = .7*up-.45*right+.8*back
                light /= np.linalg.norm(light)
                intensity = .52+.44*np.maximum(0.,normals@light)+.12*np.maximum(0.,normals@back)
                color = np.clip(intensity[:,None]*[108.,155.,175.],0,255).astype(np.uint8)
            image = np.zeros((height,width,4),dtype=np.uint8)
            code = rasterize(len(v),np.ascontiguousarray(view),len(f),f,np.ascontiguousarray(color),width,height,image)
            if code:
                raise RuntimeError(f'Invalid orthographic render ({code})')
            Image.fromarray(image).resize((1000,360),Image.Resampling.LANCZOS).save(
                prefix.with_name(f'{prefix.name}-{index}.png'))
