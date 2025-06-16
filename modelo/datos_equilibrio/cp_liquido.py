from pathlib import Path
from typing import Dict
import csv


class CPLiquido:
    """
    Capacidades caloríficas de líquidos.
    """

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "cp_liquido.csv"
        self.num_ctes = 5
        self.unidades = "J/kmol/K"

        datos = self.__leer_cp_liquido()
        self.constantes = datos["constantes"]
        self.T_min = datos["T_min"]
        self.T_max = datos["T_max"]

    def __leer_cp_liquido(self) -> dict:

        with open(self.filepath, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:

                if row["nombre"] == self.compuesto:
                    constantes = {
                        i: float(row[f"c{i}"]) for i in range(1, self.num_ctes + 1)
                    }

                    return {
                        "constantes": constantes,
                        "T_min": float(row["T_min"]),
                        "T_max": float(row["T_max"]),
                    }

        raise ValueError(
            f"Compuesto '{self.compuesto}' no encontrado en {self.filepath}"
        )

    def eval(self, T: float) -> float:
        return sum(
            self.constantes[i] * T ** (i - 1) for i in range(1, self.num_ctes + 1)
        )

    def integral(self, T: float) -> float:
        return sum(self.constantes[i] * T**i / i for i in range(1, self.num_ctes + 1))

    def integral_dT(self, T_ref: float, T: float) -> float:
        return self.integral(T) - self.integral(T_ref)


if __name__ == "__main__":
    cp = CPLiquido("propano")
    Temp = 230.0
    print(f"Cp({Temp} K) = {cp.eval(Temp)} {cp.unidades}")
