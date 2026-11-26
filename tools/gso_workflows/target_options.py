"""Study-independent command-line bindings for target and CAD receiver definitions."""
from dataclasses import replace
from fnmatch import fnmatchcase


def add_target_options(parser):
    options = (
        ('--yield-safety',('MODES','FACTOR','YIELD_MPA_OR_MATERIAL'),
         'Require a minimum yield factor on every matching case. Use material for a supplied coupon yield.'),
        ('--response-limit',('MODES','METRIC','MAXIMUM'),'Set an upper bound on a measured response.'),
        ('--minimize-response',('MODES','METRIC'),'Minimize a measured scalar within the feasible designs.'),
        ('--maximize-response',('MODES','METRIC'),'Maximize a measured scalar within the feasible designs.'),
        ('--receiver-on-fixtures',('MODES','NAME'),'Measure acceleration of the moving fixture patches.'),
        ('--receiver-on-face',('MODES','NAME','X','Y','Z','NX','NY','NZ'),
         'Measure acceleration on a CAD face selected by local point in mm and outward normal.'),
        ('--shock-transfer',('MODES','NAME','SOURCE','RECEIVER'),
         'Measure the ratio of receiver to source peak acceleration.'))
    for flag,fields,description in options:
        parser.add_argument(flag,action='append',default=[],nargs=len(fields),metavar=fields,
                            help=description+' MODES accepts case-sensitive wildcard patterns.')


def configure_targets(study, targets, args):
    from .design_targets import YieldSafety, ResponseLimit, MinimizeResponse, MaximizeResponse
    from .shock_receivers import AccelerationReceiver, ShockTransmission
    from .design_space import Face
    histories=dict(study.histories)

    def select(pattern, *, histories_only=False):
        labels=tuple(label for label in study.scenarios() if fnmatchcase(label,pattern))
        if not labels or (histories_only and set(labels)-histories.keys()):
            raise ValueError(f'Invalid scenario selection for {pattern!r}. Available: {study.scenarios()}')
        return labels

    def add_sensor(pattern, sensor, attribute):
        for label in select(pattern,histories_only=True):
            history=histories[label]
            items=getattr(history,attribute)
            if any(old.name==sensor.name for old in items):
                raise ValueError(f'Duplicate receiver or transmission {sensor.name!r} in {label}')
            histories[label]=replace(history,**{attribute:(*items,sensor)})

    for pattern,name in args.receiver_on_fixtures:
        if any(histories[label].carrier is None for label in select(pattern,histories_only=True)):
            raise ValueError('A fixture accelerometer requires a moving carrier')
        add_sensor(pattern,AccelerationReceiver(name,tuple(study.fixed_faces)),'receivers')
    for pattern,name,*coordinates in args.receiver_on_face:
        if study.reference_surface_local is None:
            raise ValueError('CAD face selection requires the imported local reference surface')
        values=[float(v) for v in coordinates]
        face=Face.pick(study.reference_surface_local,values[:3],values[3:])
        add_sensor(pattern,AccelerationReceiver(name,(face,)),'receivers')
    for pattern,name,source,receiver in args.shock_transfer:
        add_sensor(pattern,ShockTransmission(name,source,receiver),'transmissions')
    limits=list(targets.limits)
    for pattern,factor,strength in args.yield_safety:
        for label in select(pattern):
            if strength.lower()=='material':
                spec=study.case_materials.get(label)
                if spec is None or spec.tensile_yield_mpa is None:
                    raise ValueError(f'No measured yield strength assigned to {label}; supply a strength in MPa')
                value=spec.tensile_yield_mpa
                basis=' / '.join(x for x in (spec.name,spec.source,spec.assumptions) if x)
            else:
                value=float(strength)
                basis='User-supplied yield strength in MPa'
            limits.append(YieldSafety((label,),value,float(factor),basis))
    limits.extend(ResponseLimit(select(pattern),metric,float(value)) for pattern,metric,value in args.response_limit)
    goals=list(targets.extra_objectives)
    for entries,kind in ((args.minimize_response,MinimizeResponse),(args.maximize_response,MaximizeResponse)):
        goals.extend(kind(select(pattern),key) for pattern,key in entries)
    configured=replace(study,histories=histories)
    targets=replace(targets,limits=tuple(limits),extra_objectives=tuple(goals))
    targets.validate_cases(configured)
    return configured,targets
