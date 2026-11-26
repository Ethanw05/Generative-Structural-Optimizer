"""Compare measured stresses and hub shock on the same recorded service portfolio."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'build/sunbeam33/deps')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from filaments import PLA


def draw(record,output):
    parts=record['parts']
    order=[('lift','Upward lift'),('span','Horizontal along span'),('lateral','Horizontal across span'),
           ('horizontal range','Any horizontal direction'),('torque','Motor-axis torque'),('span twist','Span twist'),
           ('torque plane range','Any torque in tested plane'),('brake','Motor braking'),('drop','Oblique drop'),
           ('strike','Lateral impact')]
    series={name:{row['scenario'].removeprefix('PLA / '):row for row in item['results']} for name,item in parts.items()}
    first=next(iter(series.values()))
    active=[(key,title) for key,title in order if key in first]
    if set(key for key,_ in active)!=first.keys():
        raise ValueError('The figure needs labels for all service cases')
    if 'directional_refinement_equivalent_edge_mm' in record:
        active=[(key,title+' (base mesh)' if key.endswith(' range') and 'refinement' not in first[key] else title)
                for key,title in active]
    y=np.arange(len(active))
    fig,(stress,shock)=plt.subplots(1,2,figsize=(11.5,.54*len(active)+1.8),sharey=True,
                                    gridspec_kw={'width_ratios':[2.4,1]},layout='constrained')
    colors=('#505f72','#007d82','#c55a20')
    names={'installed':'Installed Rev 2','generated':'Generated arm','trial':'Drop-driven trial'}
    offsets=np.linspace(-.19,.19,len(parts))
    for j,(name,cases) in enumerate(series.items()):
        color=colors[j%len(colors)]
        values=[max(row['peak_MPa'] for row in (cases[k],*cases[k].get('refinement',()))) for k,_ in active]
        stress.scatter(values,y+offsets[j],s=44,color=color,zorder=3,label=names.get(name,name))
        positions=[i for i,(k,_) in enumerate(active) if 'peak_acceleration_g@hub' in cases[k]]
        shock.scatter([cases[active[i][0]]['peak_acceleration_g@hub'] for i in positions],
                      y[positions]+offsets[j],s=44,color=color,zorder=3)
    cap=PLA.tensile_yield_mpa/2.
    stress.axvline(cap,color='#9e575d',ls='--',lw=1.6,label=f'Factor 2 threshold ({cap:g} MPa)')
    stress.set_xscale('log')
    stress.set_yticks(y,[title for _,title in active]);stress.invert_yaxis()
    basis='Peak over measured meshes' if 'directional_refinement_equivalent_edge_mm' in record else 'Whole-arm peak'
    stress.set_xlabel(basis+' von Mises stress (MPa, log scale)')
    shock.set_xlabel('Peak hub acceleration (g)')
    shock.set_xlim(left=0.)
    for ax in (stress,shock):
        ax.grid(axis='both',alpha=.22,lw=.75)
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('One arm geometry across service loads',fontsize=16,fontweight='bold')
    handles,labels=stress.get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    draw(json.loads(args.report.read_text()),args.output)
