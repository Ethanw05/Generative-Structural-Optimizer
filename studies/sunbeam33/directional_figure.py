"""Render matched whole-arm stress and annotate governing elements from the signed load tests."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from directional_check import ROOT,WORK
from geometry import ROT
from gso_workflows.orthographic import color_positions,render_views
from matplotlib import colormaps
from PIL import Image,ImageDraw,ImageFont
import numpy as np


def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),size)


def draw(report,work,output,labels,additional=()):
    sample=json.loads(report.read_text())
    for path in additional:
        other=json.loads(path.read_text())
        for key in ('force_N','torque_Nmm','frame','fixtures','force_application','moment_application'):
            if other[key]!=sample[key]:
                raise ValueError('Separate studies must use the same loading and fixtures')
        if set(other['parts']) & sample['parts'].keys():
            raise ValueError('Choose distinct labels when combining part comparisons')
        sample['parts'].update(other['parts'])
    force,torque=sample['force_N'],sample['torque_Nmm']/1000.
    modes=(('span',f'{force:g} N along the arm'),('lateral',f'{force:g} N across the arm'),
           ('torque',f'{torque:g} N m about the motor axis'),('span twist',f'{torque:g} N m about the arm span'))
    cases=[(mode,label,work/label/('ABS-'+mode+' positive.npz')) for mode,_ in modes for label,_ in labels]
    rows={(mode,label):sample['parts'][label]['modes']['ABS'][mode]['positive'] for mode,label,_ in cases}
    peak=max(row['peak_MPa'] for row in rows.values())
    maximum=float(2.**np.ceil(np.log2(max(1.,peak))))
    palette=colormaps['viridis'](np.linspace(0.,1.,256))
    linear=palette.copy()
    rgb=palette[:,:3]
    linear[:,:3]=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    back=np.array([.13,-.85,.6]);back/=np.linalg.norm(back)
    right=np.array([1.,0.,0.])-back*back[0];right/=np.linalg.norm(right)
    back,right=back@ROT,right@ROT
    axes=np.stack([right,np.cross(back,right),back])
    lows,highs=[],[]
    for label,_ in labels:
        with np.load(work/label/'ABS-span positive.npz') as data:
            projected=data['v']@axes.T
            lows.append(projected.min(axis=0));highs.append(projected.max(axis=0))
    lo,hi=np.min(lows,axis=0),np.max(highs,axis=0)
    target=.5*(lo+hi)@axes
    width=1.10*max(hi[0]-lo[0],(hi[1]-lo[1])/.36)
    cfg=dict(target=[target],back=[back],right=[right],width=[width],scale=maximum,log_reference=1.,palette=linear)
    panel=660
    image=Image.new('RGB',(panel*len(labels),1220),'#f6f8f9')
    ink='#243b47'
    pen=ImageDraw.Draw(image)
    pen.text((26,18),'Horizontal loading and torsion',fill=ink,font=font(34,True))
    pen.text((26,67),'ABS  |  Elastic von Mises stress  |  Circle marks the governing element',fill=ink,font=font(20))
    for j,(_,title) in enumerate(labels):
        pen.text((panel*(j+.5),125),title,fill=ink,font=font(23,True),anchor='mm')
    with TemporaryDirectory(dir=WORK) as temp:
        base=Path(temp)
        render_views(cfg,[(path,base/str(i)) for i,(_,_,path) in enumerate(cases)],ROOT/'build/Release/gso.exe')
        for i,(mode,case_title) in enumerate(modes):
            y=148+246*i
            pen.line((25,y,image.width-25,y),fill='#ccd8dd',width=2)
            pen.text((26,y+7),case_title,fill=ink,font=font(22,True))
            for j,(label,_) in enumerate(labels):
                report_row=rows[mode,label]
                pic=Image.open(base/f'{i*len(labels)+j}-0.png').convert('RGBA')
                scaled=pic.resize((600,216),Image.Resampling.LANCZOS)
                origin=(panel*j+30,y+24)
                image.paste(scaled,origin,scaled)
                delta=np.asarray(report_row['peak_at_world_mm'])-target
                x=origin[0]+300.+600.*(delta@right)/width
                py=origin[1]+108.-600.*(delta@axes[1])/width
                pen.ellipse((x-6,py-6,x+6,py+6),outline='white',width=3)
                pen.ellipse((x-8,py-8,x+8,py+8),outline=ink,width=2)
                motion=(f"{report_row['signed_rotation_deg']:.2f} deg" if 'signed_rotation_deg' in report_row
                        else f"{report_row['signed_displacement_mm']:.3f} mm")
                pen.text((panel*(j+.5),y+224),f"{report_row['peak_MPa']:.2f} MPa   /   {motion}",
                         fill=ink,font=font(19),anchor='mm')
    y=1153
    pen.text((26,y),'MPa',font=font(20,True),fill=ink)
    left,right_edge=125,image.width-40
    for x in range(left,right_edge):
        c=rgb[int((x-left)*255/(right_edge-left))]
        pen.line((x,y,x,y+16),fill=tuple(np.rint(255*c).astype(int)))
    for tick in np.r_[0.,2.**np.arange(0,int(np.log2(maximum))+1)]:
        x=left+(right_edge-left)*color_positions(tick,maximum,1.)
        pen.text((x,y+23),f'{tick:g}',font=font(18),fill=ink,anchor='ma')
    output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('work',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--additional',type=Path,action='append',default=[])
    parser.add_argument('--column',nargs=2,action='append',metavar=('LABEL','TITLE'))
    args=parser.parse_args()
    draw(args.report,args.work,args.output,args.column or [('installed','Installed Rev 2'),('generated','Generated arm')],args.additional)
