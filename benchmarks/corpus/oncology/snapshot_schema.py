from __future__ import annotations

from pydantic import BaseModel


class OncologySnapshot(BaseModel):
    patient_name: str
    patient_identifier: str
    diagnosis: str
    stage: str
    oncologist: str
    cancer_center: str
    hemoglobin_g_dl: float
    creatinine_mg_dl: float
    primary_medication: str
