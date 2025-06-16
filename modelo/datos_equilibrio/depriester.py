import numpy as np
import csv
from pathlib import Path


def kelvin_rankine(temperatura: float) -> float:
    """
    Retorna:
        Temperatura absoluta en Rankine
    """
    return temperatura * 1.8


def paa_psia(presion: float) -> float:
    """
    Retorna:
        Presion absoluta en Psia
    """
    return presion / 1e5 * 14.5038


class DePriester:

    def __init__(self, compuesto: str):
        self.compuesto = compuesto
        self.filepath = Path(__file__).parent / "depriester.csv"

        self.parametros = self.__leer_parametros()
        self.a1 = self.parametros["a1"]
        self.a2 = self.parametros["a2"]
        self.a3 = self.parametros["a3"]
        self.b1 = self.parametros["b1"]
        self.b2 = self.parametros["b2"]
        self.b3 = self.parametros["b3"]

    def __leer_parametros(self) -> dict:

        with open(self.filepath, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:

                if row["compuesto"] == self.compuesto:
                    return {
                        "a1": float(row["a1"]),
                        "a2": float(row["a2"]),
                        "a3": float(row["a3"]),
                        "b1": float(row["b1"]),
                        "b2": float(row["b2"]),
                        "b3": float(row["b3"]),
                    }

        raise ValueError(
            f"Compuesto '{self.compuesto}' no encontrado en {self.filepath}"
        )

    def eval(
        self,
        T: float,
        P: float,
    ) -> float:
        """
        Parametros:
            T: Temperatura en Rankine
            P: Presion en Psia

        Retorna:
            Valor de K por cada componente a una temperatura y presion.
        """

        return np.exp(
            self.a1 / T**2
            + self.a2 / T
            + self.a3
            + self.b1 * np.log(P)
            + self.b2 / P**2
            + self.b3 / P
        )

    def eval_SI(
        self,
        T: float,
        P: float,
    ) -> float:
        """
        Parametros:
            T: Temperatura en Kelvin
            P: Presion en Pascales

        Retorna:
            Valor de K por cada componente a una temperatura y presion.
        """

        return self.eval(kelvin_rankine(T), paa_psia(P))


if __name__ == "__main__":
    compuesto = "etano"

    lector = DePriester(compuesto)
    K = lector.eval_SI(300.0, 101325.0)

    print(
        f"El valor de la constante K para {compuesto} a 300 K y 101325 Pa es: {K:.5f}"
    )
