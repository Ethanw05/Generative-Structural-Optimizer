"""Resolve a narrow target margin and feed the resulting spatial field into later sweeps."""
from .analysis_mesh import digest
from .design_targets import assessments


def verify_limits(item, targets, analysis):
    if not targets.limits:
        return
    if targets.verification_edges_mm and targets.verification_edges_mm[0]>=.575*analysis.analysis_pitch:
        raise ValueError('An additional target refinement must be finer than the independent fine mesh')
    item.verify_margin=True
    geometry=digest(item.mesh)
    def measure(labels,edge):
        resolution=(edge,edge/20.)
        for label in sorted(labels):
            field=analysis.field(item.mesh,geometry,label,resolution)
            item.limit_evidence.setdefault(label,[]).append(field.response)
            previous=analysis.proposal_resolution.get(label,(analysis.edge,analysis.envelope))
            if edge<previous[0]:
                analysis.proposal_resolution[label]=resolution

    labels={row['scenario'] for row in assessments(item,targets.limits)}
    for edge in targets.verification_edges_mm:
        measure(labels,edge)
    edge=min((analysis.analysis_pitch*.575,*targets.verification_edges_mm))
    for depth in range(targets.verification_refinements):
        pending=set()
        for row in assessments(item,targets.limits):
            ratios=row['utilization_by_mesh']
            disputed=not item.limit_evidence.get(row['scenario']) and min(ratios)<=1.<max(ratios)
            if row['status']=='margin_unresolved' or disputed:
                pending.add(row['scenario'])
        if not pending:
            break
        edge*=.575/.95
        measure(pending,edge)


def include_measured_fields(items,targets,analysis):
    """A finer reference computed to compare a child is also evidence about the reference."""
    if not targets.limits:
        return
    labels={label for limit in targets.limits for label in limit.cases}
    for item in items:
        identity=digest(item.mesh)
        for (key,label,(edge,_)),field in analysis.fields.items():
            if key!=identity or label not in labels or edge>=.575*analysis.analysis_pitch:
                continue
            rows=item.limit_evidence.setdefault(label,[])
            if all(row['verification_edge_mm']!=edge for row in rows):
                rows.append(field.response)
                rows.sort(key=lambda row:row['verification_edge_mm'],reverse=True)
