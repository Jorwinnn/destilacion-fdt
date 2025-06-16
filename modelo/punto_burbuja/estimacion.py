from typing import List, Callable


def residual(
    T: float,
    P: float,
    x: float,
    K: List[Callable],
):
    return sum(x_i * K_i(T, P) for x_i, K_i in zip(x, K)) - 1.0


def punto_burbuja(
    T_aprox: float,
    P: float,
    x: float,
    K: List[Callable],
) -> float:
    """
    Parámetros:
        T_aprox: Temperatura inicial
        P: Presion total
        x: Fracciones molares liquidas
        K: Funciones para calcular K por cada componente

    Retorna:
        Temperatura a la cual el liquido comienza a hervir.
    """
    from scipy.optimize import root_scalar, RootResults

    sol: RootResults = root_scalar(
        residual,
        args=(
            P,
            x,
            K,
        ),
        x0=T_aprox - 20,
        x1=T_aprox + 20,
    )

    if sol.converged:
        return sol.root
    else:
        return T_aprox
