from numpy import log
from typing import Dict

# [K]
T_critica = {
    "etano": 305.32,
    "propano": 369.83,
    "i-butano": 408.14,
    "n-butano": 425.12,
}
# [K]
T_eb_estandar = {
    "etano": 184.55,
    "propano": 231.15,
    "i-butano": 261.45,
    "n-butano": 272.15,
}
# [bar]
P_critica = {
    "etano": 48.72,
    "propano": 42.48,
    "i-butano": 36.48,
    "n-butano": 37.96,
}


def tension_superficial(
    mezcla: Dict[str, float],
    temp: float,
) -> float:
    """
    Calcula la tensión superficial [N/m] de una mezcla líquida.

    Parámetros:
        mezcla: Diccionario de los compuestos y sus fracciones molares.
        temp: Temperatura actual de la mezcla [K].
    """

    T_c = sum(T_critica[i] * mezcla[i] for i in mezcla.keys())
    P_c = sum(P_critica[i] * mezcla[i] for i in mezcla.keys())
    T_eb = sum(T_eb_estandar[i] * mezcla[i] for i in mezcla.keys())

    T_r = min(temp / T_c, 1.0)
    T_r_eb = min(T_eb / T_c, 1.0)

    a = 0.9076 * (1.0 + (T_r_eb * log(P_c / 1.013)) / (1.0 - T_r_eb))
    a = 1e-3 * (0.132 * a - 0.278)

    return a * P_c ** (2 / 3) * T_c ** (1 / 3) * (1.0 - T_r) ** (11 / 9)


def tension_articulo(compuesto: str, temp: float) -> float:
    """
    Calcula la tensión superficial según los coeficientes recomendados
    en Mulero et al. (2012) para fluidos comunes.
    Implementa etano, propano, i-butano y n-butano.
    """
    parametros = {
        "etano": {
            "coeffs": [
                {"sigma": 0.07602, "n": 1.320},
                {"sigma": -0.02912, "n": 1.676},
            ],
            "Tc": 305.322,
        },
        "propano": {
            "coeffs": [
                {"sigma": 0.05334, "n": 1.235},
                {"sigma": -0.01748, "n": 4.404},
            ],
            "Tc": 369.89,
        },
        "i-butano": {
            "coeffs": [
                {"sigma": -0.01639, "n": 2.102},
                {"sigma": 0.06121, "n": 1.304},
            ],
            "Tc": 407.81,
        },
        "n-butano": {
            "coeffs": [
                {"sigma": 0.05138, "n": 1.209},
            ],
            "Tc": 425.125,
        },
    }

    if compuesto not in parametros:
        raise ValueError(
            f"Componente '{compuesto}' no implementado en tension_articulo."
        )

    p = parametros[compuesto]
    Trel = 1 - temp / p["Tc"]
    return sum(term["sigma"] * (Trel ** term["n"]) for term in p["coeffs"])


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    compuestos = ["etano", "propano", "i-butano", "n-butano"]
    temperaturas = np.arange(135.0, 273.0, 1.0)

    tensiones_usuario = {c: [] for c in compuestos}
    tensiones_articulo = {c: [] for c in compuestos}

    for T in temperaturas:
        for comp in compuestos:
            tensiones_usuario[comp].append(tension_superficial({comp: 1.0}, temp=T))
            tensiones_articulo[comp].append(tension_articulo(comp, T))

    # Generar DataFrame y CSV
    data = {"T (K)": temperaturas}
    for comp in compuestos:
        data[f"{comp} (bb)"] = tensiones_usuario[comp]
        data[f"{comp} (mea)"] = tensiones_articulo[comp]

    df = pd.DataFrame(data)
    csv_path = "tensiones.csv"
    df.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"CSV guardado en: {csv_path}")
