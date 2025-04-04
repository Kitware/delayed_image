from numpy import ndarray
from typing import Dict
from typing import Any
from _typeshed import Incomplete

write_network_text: Incomplete


def dequantize(quant_data: ndarray, quantization: Dict[str, Any]) -> ndarray:
    ...


def quantize_float01(imdata,
                     old_min: int = ...,
                     old_max: int = ...,
                     quantize_dtype=...):
    ...


class mkslice_cls:

    def __class_getitem__(self, index):
        ...

    def __getitem__(self, index):
        ...

    def __call__(self):
        ...


mkslice: Incomplete
