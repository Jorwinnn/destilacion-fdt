from pathlib import Path
import csv


class MasaMolar:

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "masa_molar.csv"
        self.unidades = "kg/kmol"
        self.masa_molar = self.__leer_masa_molar()

    def __leer_masa_molar(self) -> dict:
        masa = {}

        with open(self.filepath, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:

                if row["nombre"] == self.compuesto:
                    masa[self.compuesto] = float(row["valor"])
                    break

        return masa

    def eval(self) -> dict:
        return self.masa_molar[self.compuesto]


if __name__ == "__main__":
    lector = MasaMolar("n-butano")
    print(lector.eval())
