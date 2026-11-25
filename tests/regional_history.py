"""Public static, harmonic and transient analyses preserve local stress across space and time."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
exe = Path(sys.argv[1]).resolve()


def invoke(cmd, case, folder):
    path = folder/(cmd+'.json')
    path.write_text(json.dumps(case))
    run = subprocess.run([exe, cmd, path],text=True,capture_output=True)
    if run.returncode not in (0,3):
        raise RuntimeError(run.stderr)
    return json.loads(run.stdout)


with tempfile.TemporaryDirectory(dir=root/'build') as scratch:
    folder = Path(scratch)
    block = json.loads((root/'examples/elastic-block.json').read_text())
    force = block['load_cases'][0]['forces']
    block['load_cases'].append(dict(name='pulse',kind='transient',end_s=.00002,max_step_s=1.e-7,
                                    time_loads=[dict(forces=force,history=[[0.,0.],[.000005,1.],[.00001,0.]])]))
    for load in block['load_cases']:
        load['recover_stress'] = True
        load['stress_reference_mm'] = [[1.,.5,.5],[3.,.5,.5]]
    cases = invoke('analyze',block,folder)['load_cases']
    for name, case in cases.items():
        peaks = case['local_stress_peaks']
        assert abs(max(p['peak_MPa'] for p in peaks)-case['peak_von_mises_MPa']) < 1.e-9
        assert abs(sum(p['basin_volume_mm3'] for p in peaks)-4.) < 1.e-9
        assert len(case['tracked_stress_peaks_MPa']) == 2
        assert max(case['tracked_stress_peaks_MPa']) == case['peak_von_mises_MPa']
        if name == 'pulse':
            assert all(0. < p['time_s'] <= .00002 for p in peaks)
    assert abs(cases['axial']['peak_von_mises_MPa']-5.) < 1.e-9

    family = json.loads((root/'examples/evolution-block.json').read_text())
    family['study']['load_cases'] = family['study']['load_cases'][:1]
    family['study']['load_cases'][0]['limits'] = {'peak_von_mises_MPa':20.}
    family['search']['evaluation_budget'] = 2
    response = invoke('evolve', family, folder)
    reference = response['stress_reference_mm']['axial']
    assert reference
    for trial in response['trials']:
        for level in trial['levels']:
            r = level['responses']['axial']
            assert len(r['tracked_stress_peaks_MPa']) == len(reference)
            assert abs(max(r['tracked_stress_peaks_MPa'])-r['peak_von_mises_MPa']) < 1.e-9
