import enum


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"
