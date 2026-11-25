"""Reference properties for comparing one fully dense printed arm. Poisson ratios are assumed."""
from gso_workflows.study_materials import IsotropicMaterial

ABS=IsotropicMaterial('ABS',2200.,.35,1040.,assumptions='Existing nominal ABS study; filament grade and orientation unknown.')
PLA=IsotropicMaterial('PLA',2300.,.35,1240.,
    'https://prusament.com/wp-content/uploads/2022/10/PLA_Prusament_TDS_2021_10_EN.pdf',
    'Prusament printed horizontal coupon, 100% rectilinear infill. Isotropic surrogate with assumed Poisson ratio 0.35.',51.)
PETG=IsotropicMaterial('PETG',1500.,.35,1270.,
    'https://prusament.com/wp-content/uploads/2022/10/PETG_Prusament_TDS_2021_10_EN.pdf',
    'Prusament printed horizontal coupon, 100% rectilinear infill. Isotropic surrogate with assumed Poisson ratio 0.35.',47.)
