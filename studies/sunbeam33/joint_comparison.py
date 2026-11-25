"""Matching underside close-ups for the geometry and verified stress of two generated parts."""
from pathlib import Path
import json
import subprocess
import sys
from geometry import WORK, ROOT, ROT, ORIGIN
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def render_joint(geometries,fields,output):
    back=np.array([.62,-.50,-.74])
    back/=np.linalg.norm(back)
    right=np.array([1.,0.,0.])-back*back[0]
    right/=np.linalg.norm(right)
    palette=colormaps['viridis'](np.linspace(0.,1.,256))
    linear=palette.copy()
    linear[:,:3]=np.where(palette[:,:3]<=.04045,palette[:,:3]/12.92,((palette[:,:3]+.055)/1.055)**2.4)
    results = [json.loads(Path(path).with_suffix('.json').read_text()) for path in fields]
    peak = max(row['loaded_face_neighborhood']['peak_MPa'] for row in results)
    scale = float(np.ceil(peak))
    camera=WORK/'joint-camera.npz'
    center=np.array([198.,0.,8.5])@ROT+ORIGIN
    np.savez(camera,target=[center],back=[back@ROT],right=[right@ROT],width=[98.],
             scale=scale,palette=linear)
    from gso_workflows.orthographic import render_views
    with np.load(camera) as views:
        render_views(views,[(p,WORK/f'joint-view-{i}') for i,p in enumerate([*geometries,*fields])],
                     ROOT/'build/Release/gso.exe')
    img=Image.new('RGB',(1400,960),'#f6f8f9');draw=ImageDraw.Draw(img)
    def font(size,bold=False):
        return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),size)
    draw.text((32,16),'Under the motor pad',fill='#162f3b',font=font(37,True))
    draw.text((1368,28),'ABS  /  33 N upward',anchor='ra',fill='#324c5a',font=font(22))
    for j,text in enumerate(('Previous generated arm','Revised generated arm')):
        draw.text((350+700*j,91),text,anchor='mm',fill='#162f3b',font=font(24,True))
    for i in range(4):
        pic=Image.open(WORK/f'joint-view-{i}-0.png').convert('RGBA')
        pic=pic.crop((150,0,850,360))
        img.paste(pic,((i%2)*700,112+(i//2)*375),pic)
    draw.line((32,480,1368,480),fill='#bdcbd1',width=2)
    for j,row in enumerate(results):
        stress = row["loaded_face_neighborhood"]["peak_MPa"]
        section = row["transfer_sections"][0]
        caption = f"Pad peak {stress:.2f} MPa   |   Effective area {section['effective_area_mm2']:.1f} mm² at {section['depth_mm']:.2f} mm"
        draw.text((350+700*j,849),caption,anchor="ma",fill="#233947",font=font(18))
    y=891
    draw.text((32,y),'von Mises (MPa)',fill='#162f3b',font=font(20,True))
    for x in range(275,1350):
        col=palette[int((x-275)*255/1074),:3]
        draw.line((x,y,x,y+15),fill=tuple((255*col).astype(int)))
    for pos in range(5):
        value = f'{scale*pos/4.:g}'
        draw.text((275+pos*1074/4,y+22),value,anchor='ma',font=font(19),fill='#162f3b')
    img.save(output)


if __name__=='__main__':
    paths=[Path(p).resolve() for p in sys.argv[1:]]
    render_joint(paths[:2],paths[2:4],paths[4])
