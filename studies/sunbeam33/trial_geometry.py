"""Export and inspect a measured arm mutation before its independent load-case checks."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import trimesh
from figures import WORK,make_figures
from gso_workflows.analysis_mesh import digest
from gso_workflows.mesh_quality import validate_single_part
from gso_workflows.surface_accuracy import sample_offsets
from gso_workflows.volume_study import publish_surface


def publish_trial_geometry(parent,trial,work,output,driver,window,spacing):
    detail=output/'Drop-trial'
    detail.mkdir(parents=True,exist_ok=True)
    parts={}
    for name,surface in (('parent',parent),('trial',trial)):
        validate_single_part(surface)
        parts[name]=dict(geometry=digest(surface),volume_mm3=float(surface.volume),
                         triangles=len(surface.faces),surface_area_mm2=float(surface.area))
    destination=output/'Front-Left-drop-trial.obj'
    with TemporaryDirectory(dir=work) as temp:
        staged=Path(temp)
        obj=staged/destination.name
        publish_surface(trial,obj,staged)
        imported=trimesh.load_mesh(obj,process=False,maintain_order=True)
        np.testing.assert_array_equal(imported.faces,trial.faces)
        tolerance=5.e-14+np.finfo(float).eps*np.max(np.abs(trial.vertices))
        np.testing.assert_allclose(imported.vertices,trial.vertices,rtol=0.,atol=tolerance)
        parts['trial']['max_export_roundoff_mm']=float(np.max(np.abs(imported.vertices-trial.vertices)))
        parts['trial']['obj_lf_sha256']=hashlib.sha256(obj.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        obj.replace(destination)
    results={}
    for i,(name,src,reference) in enumerate((('trial_to_parent',trial,parent),('parent_to_trial',parent,trial))):
        seed=293+i
        offsets=sample_offsets(src,reference,30000,seed)
        absolute=np.abs(offsets)
        results[name]=dict(area_samples=len(offsets),seed=seed,absolute_median_mm=float(np.median(absolute)),
                           absolute_p95_mm=float(np.quantile(absolute,.95)),absolute_p99_mm=float(np.quantile(absolute,.99)),
                           signed_p05_mm=float(np.quantile(offsets,.05)),signed_p95_mm=float(np.quantile(offsets,.95)),
                           sampled_fraction_over_0_25_mm=float(np.mean(absolute>.25)),
                           sampled_fraction_over_1_mm=float(np.mean(absolute>1.)))
    record=dict(parts=parts,driving_case=driver,field_window_s=window,boundary_sampling_mm=spacing,sampled_boundary_change=results)
    (detail/'Geometry.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    make_figures(work/'candidate.npz',WORK/'native-arm.npz',None,None,detail,'Geometry','Drop-driven trial   /   equal material budget')
