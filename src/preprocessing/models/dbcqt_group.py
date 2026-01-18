from pydantic import BaseModel, ConfigDict
from numpy import ndarray, float32

class DBCQTDimensionMatchError(Exception):
    ...

class DBCQTGroup(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    cqts: list[ndarray]
    song_path: str
    overall_max: float32

    @staticmethod
    def for_path(path: str) -> "DBCQTGroup":
        return DBCQTGroup(
            cqts=[],
            song_path=path,
        )
    
    def add(self, cqt: ndarray) -> None:
        if not self.cqt_count == 0 and self.cqts[-1].shape != cqt.shape:
            raise DBCQTDimensionMatchError(
                "The dimensions of the Db CQT to be added do not match that of the ones which already exist."
                + f"Expected dimensions {self.cqts[-1].shape}. Given dimensions {cqt.shape}. Number of cqts so far: {self.cqt_count}"
            )
        self.cqts.append(cqt)

    @property
    def cqt_count(self):
        return len(self.cqts)