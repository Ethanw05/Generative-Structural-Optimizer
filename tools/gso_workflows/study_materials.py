"""Apply alternative material specifications to the same CAD geometry and service cases."""
from dataclasses import dataclass,replace
import math


@dataclass(frozen=True)
class IsotropicMaterial:
    name: str
    young_mpa: float
    poisson: float
    density_kg_m3: float
    source: str = ''
    assumptions: str = ''
    tensile_yield_mpa: float | None = None  # Comparison datum, separate from the constitutive calibration.

    def __post_init__(self):
        if not self.name or not all(math.isfinite(x) for x in (self.young_mpa,self.poisson,self.density_kg_m3)):
            raise ValueError('A material needs a name and finite properties')
        if self.young_mpa<=0. or not -1.<self.poisson<.5 or self.density_kg_m3<=0.:
            raise ValueError('Invalid isotropic elastic properties or mass density')
        if self.tensile_yield_mpa is not None and not (math.isfinite(self.tensile_yield_mpa) and self.tensile_yield_mpa>0.):
            raise ValueError('Invalid measured tensile yield strength')

    def apply(self,study):
        return replace(study,young_mpa=self.young_mpa,poisson=self.poisson,density_kg_m3=self.density_kg_m3)


def material_portfolio(study,materials):
    """Give each material/load pair a distinct mechanical field and the same evolving solid."""
    materials=tuple(materials)
    if not materials or len({m.name for m in materials})!=len(materials):
        raise ValueError('A material portfolio requires unique material names')
    loads={};histories={};specs={};laws={}
    for material in materials:
        for label in study.scenarios():
            key=f'{material.name} / {label}'
            if key in specs:raise ValueError('Ambiguous material and case names')
            destination,source=(loads,study.loads) if label in study.loads else (histories,study.histories)
            destination[key]=source[label]
            specs[key]=material
            if label in study.solid_laws:laws[key]=study.solid_laws[label]
    return replace(study,loads=loads,histories=histories,case_materials=specs,solid_laws=laws)
