from pathlib import Path
import csv


class CPVapor:
    """
    Capacidades calorificas de vapores.
    """

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "cp_vapor.csv"
        self.num_ctes = 4
        self.unidades = "J/kmol/K"

        datos = self.__leer_cp_vapor()
        self.constantes = datos["constantes"]
        self.T_min: float = datos["T_min"]
        self.T_max: float = datos["T_max"]

    def __leer_cp_vapor(self) -> dict:

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
        return 1000 * sum(
            self.constantes[i] * T ** (i - 1) for i in range(1, self.num_ctes + 1)
        )

    def integral(self, T: float) -> float:
        return 1000 * sum(
            self.constantes[i] * T**i / i for i in range(1, self.num_ctes + 1)
        )

    def integral_dT(self, T_ref: float, T: float) -> float:
        return self.integral(T) - self.integral(T_ref)


if __name__ == "__main__":
    cp = CPVapor("propano")
    Temp = 300.0
    print(f"Cp({Temp} K) = {cp.eval(Temp)} {cp.unidades}")
