"""Same cameras and quantitative color scale for the generated and installed arm."""
from geometry import WORK,ROOT,ROT,ORIGIN
from pathlib import Path
import numpy as np
import json,sys
from tempfile import TemporaryDirectory
from PIL import Image,ImageDraw,ImageFont
from matplotlib import colormaps


def make_figures(generated_geometry,installed_geometry,generated_stress,installed_stress,out,mesh_caption='Fine mesh'):
    with TemporaryDirectory(dir=WORK) as directory:
        _render(generated_geometry,installed_geometry,generated_stress,installed_stress,out,mesh_caption,Path(directory))


def _render(generated_geometry,installed_geometry,generated_stress,installed_stress,out,mesh_caption,stage):
    out.mkdir(parents=True,exist_ok=True)
    geometries=[np.load(p)['v'] for p in (generated_geometry,installed_geometry)]
    v=(np.vstack(geometries)-ORIGIN)@ROT.T
    pad=(np.array([-152.4,0,161.925])-ORIGIN)@ROT.T
    views=[('Oblique',(.17,-1.,.58),None),('Top',(0.,0.,1.),None),
           ('Side',(0.,-1.,0.),None),('Under the motor pad',(.62,-.50,-.74),pad+[-8,0,-8])]
    stress_files=(generated_stress,installed_stress) if generated_stress is not None else ()
    results=[json.loads(Path(p).with_suffix('.json').read_text()) for p in stress_files]
    have_peaks=bool(results) and all('peak_at_world_mm' in result for result in results)
    if have_peaks:
        peaks=np.array([row['peak_at_world_mm'] for row in results])
        center=(peaks.mean(axis=0)-ORIGIN)@ROT.T
        views.append(('Governing stress location (ring marks the projected element center)',(.10,-1.,.12),center))
    target=[];directions=[];rights=[];widths=[]
    for _,back,center in views:
        back=np.asarray(back);back/=np.linalg.norm(back)
        right=np.array([1.,0.,0.])-back*back[0];right/=np.linalg.norm(right)
        up=np.cross(back,right)
        if center is None:
            axes=np.array([right,up,back])
            projected=v@axes.T;lo,hi=projected.min(0),projected.max(0)
            center=.5*(lo+hi)@axes
            width=max(hi[0]-lo[0],(hi[1]-lo[1])*1000/360)*1.08
        else:width=110.
        target.append(center@ROT+ORIGIN);directions.append(back@ROT);rights.append(right@ROT);widths.append(width)
    # Round up to a power of two so the shared scale includes every measured element.
    peak=max((row['peak_MPa'] for row in results),default=1.)
    maximum=float(2.**np.ceil(np.log2(max(1.,peak))))
    log_reference=1.
    color=colormaps['viridis'](np.linspace(0,1,256))
    linear=color.copy();linear[:,:3]=np.where(color[:,:3]<=.04045,color[:,:3]/12.92,((color[:,:3]+.055)/1.055)**2.4)
    cfg=stage/'inspection-cameras.npz'
    np.savez(cfg,target=target,back=directions,right=rights,width=widths,scale=maximum,log_reference=log_reference,palette=linear)
    from gso_workflows.orthographic import render_views,color_positions
    inputs=(generated_geometry,installed_geometry,*stress_files)
    with np.load(cfg) as camera:
        render_views(camera, [(p,stage/f'inspect-{j}') for j,p in enumerate(inputs)], ROOT/'build/Release/gso.exe')
    def font(size,bold=False):
        return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),size)
    for field in ((False,True) if stress_files else (False,)):
        rows=[0,1,2,3] if not field else ([0,2,3,4] if have_peaks else [0,2,3])
        height=145+len(rows)*330+(88 if field else 10)
        canvas=Image.new('RGB',(1680,height),'#f6f8f9');d=ImageDraw.Draw(canvas)
        title='Elastic von Mises stress' if field else 'Front-left arm geometry'
        d.text((42,20),title,font=font(40,True),fill='#1c2e3a')
        d.text((1638,36),'ABS   /   33 N upward',font=font(24),fill='#314957',anchor='ra')
        for j,name in enumerate(['Generated','Installed Rev 2']):
            x=420+840*j
            d.text((x,92 if field else 102),name,font=font(27,True),fill='#243e4c',anchor='mm')
            if field:
                report=results[j]
                pad_peak=report['loaded_face_neighborhood']['peak_MPa']
                d.text((x,125),f"{mesh_caption}   ·   Pad {pad_peak:.2f} MPa   ·   Whole arm {report['peak_MPa']:.2f} MPa",
                       font=font(19),fill='#314957',anchor='mm')
        for i,row in enumerate(rows):
            y=145+330*i
            d.line((42,y,1638,y),fill='#ced8dc',width=2)
            d.text((42,y+7),views[row][0],font=font(21),fill='#3e5866')
            for j in range(2):
                pic=Image.open(stage/f'inspect-{(2 if field else 0)+j}-{row}.png').convert('RGBA')
                pic=pic.resize((820,295),Image.Resampling.LANCZOS)
                canvas.paste(pic,(10+840*j,y+29),pic)
                if field and row==4:
                    delta=np.asarray(results[j]['peak_at_world_mm'])-target[row]
                    x=420+840*j+(delta@rights[row])*820/widths[row]
                    py=y+29+147.5-(delta@np.cross(directions[row],rights[row]))*295/(.36*widths[row])
                    d.ellipse((x-10,py-10,x+10,py+10),outline='white',width=5)
                    d.ellipse((x-12,py-12,x+12,py+12),outline='#21343e',width=2)
        if field:
            y=height-77
            d.text((42,y),'von Mises (MPa)',font=font(22,True),fill='#233947')
            d.text((42,y+31),'Log(1 + stress / 1 MPa)',font=font(16),fill='#314957')
            x0,x1=310,1600
            for x in range(x0,x1):
                c=color[int((x-x0)*255/(x1-x0)),:3]
                d.line((x,y,x,y+17),fill=tuple((255*c).astype(int)))
            for value in np.r_[0.,2.**np.arange(0,int(np.log2(maximum))+1)]:
                q=color_positions(value,maximum,log_reference)
                label=f'{value:g}'
                d.text((x0+(x1-x0)*q,y+22),label,anchor='ma',font=font(21),fill='#233947')
        canvas.save(out/('Stress-comparison.png' if field else 'Geometry-comparison.png'))


if __name__=='__main__':make_figures(*[Path(p).resolve() for p in sys.argv[1:]])
