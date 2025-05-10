import enum


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class CreditRiskType(str, enum.Enum):
    good = "Good"
    standard = "Standard"
    poor = "Poor"
