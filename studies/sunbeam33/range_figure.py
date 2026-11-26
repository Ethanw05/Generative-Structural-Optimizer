"""Plot whole-solid directional stress envelopes from verified linear load-plane studies."""
import argparse
import json
from pathlib import Path
from directional_check import PLA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(reports,parts,output):
    designs={}
    amplitudes=None
    for path in reports:
        record=json.loads(path.read_text())
        values=(record['force_N'],record['torque_Nmm'])
        if amplitudes is not None and amplitudes!=values:
            raise ValueError('Range plots require identical force and torque amplitudes')
        amplitudes=values
        if designs.keys() & record['parts'].keys():
            raise ValueError('Duplicate part names in range studies')
        designs.update(record['parts'])
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),sharey=True,layout='constrained')
    colors=('#596574','#007a84','#c65b20')
    for ax,(key,title,axis) in zip(axes,(('horizontal',f'{amplitudes[0]:g} N horizontal force','Span toward lateral (degrees)'),
                                        ('torque plane',f'{amplitudes[1]/1000:g} N m bolt-group moment','Motor axis toward span (degrees)')),strict=True):
        cap=PLA.tensile_yield_mpa/2.
        ax.axhline(cap,color='#985158',lw=1.5,ls='--',label=f'PLA factor 2 threshold ({cap:g} MPa)')
        for i,(name,label) in enumerate(parts):
            row=designs[name]['planes'][key]
            color=colors[i%len(colors)]
            ax.plot(np.arange(181),row['peak_curve_MPa'],label=label,color=color,lw=2.5)
            ax.scatter(row['stress_angle_deg'],row['direct_peak_MPa'],color=color,edgecolor='white',s=60,zorder=4)
        ax.set(xlabel=axis,xlim=(0,180),xticks=[0,30,60,90,120,150,180],title=title)
        ax.grid(axis='y',lw=.6,alpha=.3)
        for spine in ('top','right'):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel('Whole-arm peak von Mises stress (MPa)')
    axes[0].set_ylim(bottom=0.)
    axes[1].legend(loc='lower right',frameon=False,fontsize=8)
    fig.suptitle('Directional stress through a full reversal',fontsize=16,fontweight='bold')
    fig.supxlabel('Markers show the independent solves at the maximum. The stress repeats every 180 degrees.',fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('reports',nargs='+',type=Path)
    cli.add_argument('--part',nargs=2,action='append',metavar=('LABEL','TITLE'))
    args=cli.parse_args()
    draw(args.reports,args.part or [('installed','Installed Rev 2'),('generated','Generated arm')],args.output)
