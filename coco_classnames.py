from enum import Enum


class Classnames(Enum):
    vehicles = {
        'CAR': 2,
        'MOTORBIKE': 3,
        'BUS': 5,
        'TRUCK': 7,
    }

    @classmethod
    def get_vehicles(cls):
        return [*cls.vehicles.value.values()]
