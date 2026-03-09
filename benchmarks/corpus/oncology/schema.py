from __future__ import annotations

from pydantic import BaseModel


class TextCode(BaseModel):
    text: str


class ValueQuantity(BaseModel):
    value: float
    unit: str


class Patient(BaseModel):
    resourceType: str
    identifier: str
    name: str
    gender: str
    birthDate: str


class Encounter(BaseModel):
    resourceType: str
    periodStart: str
    participant: str
    serviceProvider: str


class Condition(BaseModel):
    resourceType: str
    code: TextCode
    clinicalStatus: str
    stage: str
    bodySite: str


class DiagnosticReport(BaseModel):
    resourceType: str
    code: TextCode
    effectiveDateTime: str
    conclusion: str


class Observation(BaseModel):
    resourceType: str
    code: TextCode
    effectiveDateTime: str
    valueQuantity: ValueQuantity


class MedicationRequest(BaseModel):
    resourceType: str
    status: str
    medicationCodeableConcept: TextCode
    dosageInstruction: str
    route: str


class CarePlan(BaseModel):
    resourceType: str
    status: str
    intent: str
    description: str


class OncologyBundle(BaseModel):
    resourceType: str
    type: str
    patient: Patient
    encounter: Encounter
    condition: Condition
    diagnosticReport: DiagnosticReport
    observations: list[Observation]
    medications: list[MedicationRequest]
    carePlan: CarePlan
