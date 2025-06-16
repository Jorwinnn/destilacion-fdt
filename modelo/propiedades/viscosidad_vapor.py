from pathlib import Path
import csv


class ViscosidadVapor:
    """
    Viscosidades de vapores.
    """

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "viscosidad_vapor.csv"
        self.num_ctes = 3
        self.unidades = "Pa s"

        datos = self.__leer_viscosidad_vapor()
        self.constantes = datos["constantes"]
        self.T_min = datos["T_min"]
        self.T_max = datos["T_max"]

    def __leer_viscosidad_vapor(self) -> dict:

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


if __name__ == "__main__":
    visc = ViscosidadVapor("propano")
    Temp = 350.0
    print(f"Viscosidad({Temp} K) = {visc.eval(Temp)} {visc.unidades}")
