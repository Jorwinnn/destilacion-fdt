from pathlib import Path
import csv


class DHVap:
    """
    Calores de vaporización.
    """

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "calor_vaporizacion.csv"
        self.unidades = "J/kmol"

        datos = self.__leer_dhvap()
        self.valor: float = datos["valor"]
        self.T_ref: float = datos["T_ref"]

    def __leer_dhvap(self) -> dict:

        with open(self.filepath, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:

                if row["nombre"] == self.compuesto:
                    return {
                        "valor": float(row["valor"]),
                        "T_ref": float(row["T_ref"]),
                    }

        raise ValueError(
            f"Compuesto '{self.compuesto}' no encontrado en {self.filepath}"
        )

    def eval(self) -> float:
        return self.valor


if __name__ == "__main__":
    dh = DHVap("n-butano")
    print(f"ΔHvap: {dh.eval()} {dh.unidades}")
    print(f"T_ref: {dh.T_ref} K")
