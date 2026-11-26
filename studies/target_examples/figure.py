"""Draw the actual original and selected OBJs and their common-load fine FE stress fields."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'build/sunbeam33/deps')]
import numpy as np
import trimesh
from matplotlib import colormaps
from PIL import Image,ImageDraw,ImageFont
from gso_workflows.orthographic import render_views


def font(size, bold=False):
    for regular,heavy in [('C:/Windows/Fonts/segoeui.ttf','C:/Windows/Fonts/segoeuib.ttf'),
                          ('DejaVuSans.ttf','DejaVuSans-Bold.ttf')]:
        try:
            return ImageFont.truetype(heavy if bold else regular,size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_figure(case,executable):
    out=ROOT/'runs/Target-studies'/case
    work=ROOT/'build/target-studies'/case
    selection=json.loads((work/'generated-surface/selection.json').read_text())
    result=json.loads((out/'comparison.json').read_text())
    run=Path(selection['feedback_directory'])
    bounds=[]
    shapes=[]
    columns=(('original','initial'),('generated',selection['selected']))
    for name,tag in columns:
        model=trimesh.load_mesh(out/(name+'.obj'),process=False)
        bounds.append(model.vertices)
        surface=work/(name+'-figure.npz')
        np.savez(surface,v=model.vertices,f=model.faces)
        shapes.append((surface,work/name))
    vertices=np.vstack(bounds)
    cfg=dict(target=[],back=[],right=[],width=[])
    for direction in ((1.,-1.8,1.1),(1.,-1.8,-1.1)):
        camera=np.array(direction)
        camera/=np.linalg.norm(camera)
        right=np.cross([0.,0.,1.],camera)
        right/=np.linalg.norm(right)
        axes=np.array([right,np.cross(camera,right),camera])
        projected=vertices@axes.T
        span=np.ptp(projected,axis=0)
        cfg['target'].append(.5*(projected.max(0)+projected.min(0))@axes)
        cfg['back'].append(camera)
        cfg['right'].append(right)
        cfg['width'].append(max(span[0],span[1]*1000./360.)*1.12)
    color=colormaps['viridis'](np.linspace(0.,1.,256))
    palette=color.copy()
    palette[:,:3]=np.where(color[:,:3]<=.04045,color[:,:3]/12.92,((color[:,:3]+.055)/1.055)**2.4)
    cfg['palette']=palette
    render_views(cfg,shapes,executable)
    fine=[result['comparisons'][name]['responses'][1] for name,_ in columns]
    canvas=Image.new('RGB',(1120,460+410*len(fine[0])),'#f4f7f8')
    draw=ImageDraw.Draw(canvas)
    title={'necked-link':'Necked tension link','impact-mount':'Payload impact mount'}[case]
    draw.text((28,14),title,font=font(34,True),fill='#152d38')
    draw.text((28,65),'Equal material budget   /   same loads and fixtures   /   assumed 30 MPa yield',font=font(21),fill='#375461')
    for column,(name,_) in enumerate(columns):
        x=280+560*column
        draw.text((x,122),('Original','Solver selection')[column],anchor='mm',font=font(24,True),fill='#1a3542')
        checks=result['comparisons'][name]['checks']
        yield_checks=[check for check in checks if check['quantity']=='whole-part yield factor of safety']
        governing=min(yield_checks,key=lambda check:check['achieved'])
        reason='  (mesh margin unresolved)' if governing.get('status')=='margin_unresolved' else ''
        draw.text((x,153),f"Minimum sampled yield factor {governing['achieved']:.2f}  /  target {governing['minimum']:g}"+reason,
                  anchor='mm',font=font(16),fill='#375461')
        for j,view in enumerate(('Oblique','Underside')):
            with Image.open(work/f'{name}-{j}.png') as image:
                crop=image.crop((246,0,754,360)).resize((270,191),Image.Resampling.LANCZOS)
                canvas.paste(crop,(x-272+272*j,168),crop)
            draw.text((x-137+272*j,365),view,anchor='mm',font=font(18),fill='#375461')
    for index,record in enumerate(fine[0]):
        start=400+index*410
        scenario=record['scenario']
        fields=[(run/f'{tag}-level1-case{index}.npz',work/f'{name}-stress-{index}') for name,tag in columns]
        peak=max(rows[index]['peak_MPa'] for rows in fine)
        scale=10.**np.floor(np.log10(peak))
        cfg['scale']=float(np.ceil(peak/scale*4.)/4.*scale)
        render_views(cfg,fields,executable)
        draw.line((28,start,1092,start),fill='#bdcdd1',width=2)
        draw.text((28,start+6),scenario.capitalize()+'  |  von Mises stress',font=font(21,True),fill='#253e4d')
        for column,(name,_) in enumerate(columns):
            x=280+560*column
            with Image.open(work/f'{name}-stress-{index}-0.png') as image:
                crop=image.crop((220,0,780,360)).resize((448,288),Image.Resampling.LANCZOS)
                canvas.paste(crop,(x-224,start+43),crop)
            row=fine[column][index]
            metric=f"Peak {row['peak_MPa']:.2f} MPa"
            if 'peak_acceleration_g@payload' in row:
                metric+=f"   ·   Payload {row['peak_acceleration_g@payload']:.3f} g"
            elif 'compliance_Nmm' in row:
                metric+=f"   ·   Compliance {row['compliance_Nmm']:.3f} N mm"
            draw.text((x,start+349),metric,font=font(18),anchor='mm',fill='#253e4d')
        for i in range(560):
            rgb=tuple((255*color[int(i*255/560),:3]).astype(int))
            draw.line((280+i,start+376,280+i,start+387),fill=rgb)
        draw.text((269,start+382),'0',anchor='rm',font=font(15),fill='#253e4d')
        draw.text((853,start+382),f"{cfg['scale']:g} MPa",anchor='lm',font=font(15),fill='#253e4d')
    draw.text((28,canvas.height-32),'Stress maps show the fine boundary mesh. Safety margins include independent target verification.',
              font=font(17),fill='#375461')
    canvas.save(out/'comparison.png')


if __name__=='__main__':
    make_figure(sys.argv[1],Path(sys.argv[2]).resolve())
