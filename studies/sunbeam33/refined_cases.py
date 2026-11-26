"""Attach matched finer directional tests to the same solid and physical load definitions."""
import numpy as np
from filaments import ABS,PLA


def refined_cases(identity,base_reports,axis_reports,angle_reports):
    def find(reports):
        matches=[(report,part) for report in reports for part in report['parts'].values()
                 if part['geometry']==identity]
        if len(matches)!=1:
            raise ValueError(f'Expected one record for geometry {identity}, got {len(matches)}')
        return matches[0]

    if not axis_reports and not angle_reports:
        return {},None
    coarse,part=find(base_reports)
    axes,checked=find(axis_reports)
    for key in ('force_N','torque_Nmm','fixtures','frame','moment_application','force_application','linear_elastic'):
        if axes[key]!=coarse[key]:
            raise ValueError('Refinement changed the study input '+key)
    ref=checked['mesh']['refinement']
    if ref['source_tetrahedra']!=part['n_tets'] or checked['n_tets']<=part['n_tets']:
        raise ValueError('Refined cells must derive from the compared analysis boundary')
    if abs(ABS.poisson-PLA.poisson)>1.e-12 or not axes['linear_elastic']:
        raise ValueError('This stress transfer requires linear isotropic materials with a common Poisson ratio')
    if checked['modes']['ABS'].keys()!=part['modes']['PLA'].keys():
        raise ValueError('Missing signed load modes in the refinement')
    scale=ABS.young_mpa/PLA.young_mpa
    evidence={}
    for name,pair in checked['modes']['ABS'].items():
        label='PLA / '+name
        motion='signed_rotation_deg' if 'signed_rotation_deg' in pair['positive'] else 'signed_displacement_mm'
        factors={'peak_MPa':1.,'regional_peak_rms_MPa':1.,'compliance_Nmm':scale,motion:scale}
        for key,factor in factors.items():
            for sign in ('positive','negative'):
                absolute=part['modes']['ABS'][name][sign][key]
                tested=part['modes']['PLA'][name][sign][key]
                np.testing.assert_allclose(absolute*factor,tested,rtol=1.e-7,atol=1.e-10)
            symmetry=-1. if key==motion else 1.
            np.testing.assert_allclose(pair['negative'][key],symmetry*pair['positive'][key],rtol=1.e-7)
        evidence[label]=[dict(scenario=label,n_tets=checked['n_tets'],
                              **{key:factor*pair['positive'][key] for key,factor in factors.items()})]
    if angle_reports:
        report,angles=find(angle_reports)
        if (report['material']!='PLA' or any(report[key]!=axes[key] for key in ('force_N','torque_Nmm'))
            or angles['n_tets']!=checked['n_tets'] or angles['mesh']['refinement']!=ref):
            raise ValueError('Angled and signed tests require the same refined mesh and loading amplitudes')
        for family,row in angles['planes'].items():
            label='PLA / '+family+' range'
            evidence[label]=[dict(scenario=label,n_tets=angles['n_tets'],**{key:row[key] for key in
                ('peak_MPa','maximum_compliance_Nmm','regional_peak_rms_at_stress_angle_MPa','stress_angle_deg','compliance_angle_deg')})]
    return evidence,ref['regular_tet_equivalent_edge_mm']
