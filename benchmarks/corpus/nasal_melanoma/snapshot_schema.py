from __future__ import annotations

from pydantic import BaseModel


class NasalMelanomaSnapshot(BaseModel):
    age_years: int
    sex: str
    comorbidities: list[str]
    presenting_symptom_month: str
    primary_site: str
    diagnosis: str
    ct_sinus_date: str
    ct_lesion_size_cm: list[float]
    mri_face_date: str
    mri_mass_size_cm: list[float]
    left_submandibular_node_cm: float
    pet_ct_date: str
    pet_hypermetabolic_nasal_mass: bool
    metastatic_disease_on_pet: bool
    left_apical_lung_nodule_present: bool
    surgery_date: str
    surgery_procedures: list[str]
    pathology_size_cm: float
    margins_negative: bool
    lymphovascular_invasion: bool
    perineural_invasion: bool
    nodes_positive: int
    nodes_examined: int
    pathologic_t_stage: str
    pathologic_n_stage: str
