from pathlib import Path
from typing import Dict
import csv


class DensidadLiquido:

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "densidad_liquido.csv"
        self.unidades = "kg/m3"
        self.densidad = self.__leer_densidad()

    def __leer_densidad(self) -> dict:
        densidad = {}

        with open(self.filepath, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:

                if row["nombre"] == self.compuesto:
                    densidad[self.compuesto] = float(row["valor"])
                    break

        return densidad

    def eval(self) -> dict:
        return self.densidad[self.compuesto]


if __name__ == "__main__":
    lector = DensidadLiquido("n-butano")
    print(lector.eval())
