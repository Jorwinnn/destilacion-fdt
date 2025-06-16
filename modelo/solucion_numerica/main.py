from modelo.tridiagonal.solucionador import resolver_diagonal
from modelo.propiedades.criticas import tension_superficial
from modelo.punto_burbuja.estimacion import punto_burbuja
from scipy.sparse import linalg, diags
from scipy.optimize import fsolve
import numpy as np

from modelo.datos_equilibrio.depriester import DePriester
from modelo.datos_equilibrio.cp_liquido import CPLiquido
from modelo.datos_equilibrio.cp_vapor import CPVapor
from modelo.datos_equilibrio.calor_vaporizacion import DHVap
from modelo.propiedades.masa_molar import MasaMolar
from modelo.propiedades.densidad_liquido import DensidadLiquido
from modelo.propiedades.viscosidad_vapor import ViscosidadVapor

from typing import List, Dict
from pprint import pprint


class Modelo:

    def __init__(
        self,
        componentes: list = None,
        z: list = None,
        N: int = None,
        etp_al: int = None,
        F: float = None,
        D: float = None,
        R: float = None,
        T_al_apx: float = None,
        P: float = None,
        dtro: float = None,
    ):
        self.componentes: List[str] = componentes
        self.z_al: Dict[str, float] = {i: x for i, x in zip(componentes, z)}
        self.N: int = N
        self.etp_al: int = etp_al
        self.F_al: float = F
        self.D: float = D
        self.B: float = F - D
        self.R: float = R
        self.T_al: float = T_al_apx
        self.P_al: float = P

        self.tol_temperatura: float = 1.0e-2
        self.tol_presion: float = 1.0e1
        self.tol_densidad: float = 1.0e-3
        self.tol_velocidad: float = 1.0e-3
        self.tol_caudal: float = 1.0e-4

        self.K_func: Dict[str, DePriester] = {}
        self.cp_liquido_func: Dict[str, CPLiquido] = {}
        self.cp_vapor_func: Dict[str, CPVapor] = {}
        self.dH_func: Dict[str, DHVap] = {}
        self.T_ref: Dict[str, float] = {}
        self.denL_est: Dict[str, DensidadLiquido] = {}
        self.masa_molar: Dict[str, MasaMolar] = {}
        self.visc_vapor_func: Dict[str, ViscosidadVapor] = {}

        self.num_platos: float = self.N + 1
        self.etapas = np.arange(self.num_platos)
        self.F = np.zeros(self.num_platos)
        self.L = np.zeros(self.num_platos)
        self.V = np.zeros(self.num_platos)
        self.L_ant = np.zeros(self.num_platos)
        self.V_ant = np.zeros(self.num_platos)
        self.z = {i: np.zeros(self.num_platos) for i in componentes}
        self.l = {i: np.zeros(self.num_platos) for i in componentes}
        self.F[self.etp_al] = self.F_al

        for i in self.componentes:
            self.z[i][self.etp_al] = self.z_al[i]

        self.T = np.zeros(self.num_platos)
        self.T_ant = np.zeros(self.num_platos)
        self.P = np.zeros(self.num_platos)
        self.P_ant = np.zeros(self.num_platos)
        self.dP = np.zeros(self.num_platos)
        self.K = {i: np.zeros(self.num_platos) for i in self.componentes}

        self.denL = np.zeros(self.num_platos)
        self.denV = np.zeros(self.num_platos)
        self.u = np.zeros(self.num_platos)
        self.u_min = np.zeros(self.num_platos)
        self.u_max = np.zeros(self.num_platos)

        self.alpha: float = 0.3  # amortiguacion de temperatura
        self.beta: float = 0.3  # amortiguacion de presion
        self.iter_internas = 10

        self.grav = 9.81
        self.S = None
        self.alt_liq = 0.1
        self.dtro = dtro
        self.dtro_h = self.dtro * 2 / 100
        self.area = np.pi * self.dtro**2 / 4
        self.area_ac = self.area * 63.66 / 100
        self.area_hs = self.area_ac * 20 / 100
        self.num_hs = int(self.area_hs / (np.pi * self.dtro_h**2 / 4))

    def error_relativo_minimo(self, nuevo, viejo, tol):
        return np.abs((nuevo - viejo) / (nuevo + 1e-10)).max() < tol

    def temperaturas_estables(self):
        return np.abs(self.T - self.T_ant).max() < self.tol_temperatura

    def presiones_estables(self):
        return np.abs(self.P - self.P_ant).max() < self.tol_presion

    def agregar_params(self):

        for i in self.componentes:
            self.K_func[i] = DePriester(i)
            self.cp_liquido_func[i] = CPLiquido(i)
            self.cp_vapor_func[i] = CPVapor(i)
            self.dH_func[i] = DHVap(i)
            self.masa_molar[i] = MasaMolar(i)
            self.denL_est[i] = DensidadLiquido(i)
            self.visc_vapor_func[i] = ViscosidadVapor(i)

        self.T_ref = {key: val.T_ref for key, val in self.dH_func.items()}

    def xij(self, i, j):
        return self.l[i][j] / self.L[j]

    def h_pura(self, i, T):
        return self.cp_liquido_func[i].integral_dT(self.T_ref[i], T)

    def hj(self, j):
        return sum(self.xij(i, j) * self.h_pura(i, self.T[j]) for i in self.componentes)

    def h_F(self, j):
        return sum(self.z[i][j] * self.h_pura(i, self.T_al) for i in self.componentes)

    def yij(self, i, j):
        return self.K_func[i].eval_SI(self.T[j], self.P[j]) * self.xij(i, j)

    def H_pura(self, i, T):
        return (
            self.cp_vapor_func[i].integral_dT(self.T_ref[i], T) + self.dH_func[i].eval()
        )

    def Hj(self, j):
        return sum(self.yij(i, j) * self.H_pura(i, self.T[j]) for i in self.componentes)

    def resolver_bal_masa_comp(self, i):
        A, B, C, D = crear_abc(
            self.N,
            self.F,
            self.z[i],
            self.D,
            self.B,
            self.L,
            self.V,
            self.K[i],
        )
        self.l[i][:] = resolver_diagonal(A, B, C, D)

    def Q_condensador(self):
        return self.D * (1.0 + self.R) * (self.hj(0) - self.Hj(1))

    def Q_rehervidor(self):
        return (
            self.D * self.hj(0)
            + self.B * self.hj(self.N)
            - self.F_al * self.h_F(self.etp_al)
            - self.Q_condensador()
        )

    def resolver_bal_energia(self):
        self.L_ant[:] = self.L[:]
        self.V_ant[:] = self.V[:]

        BE = np.zeros(self.num_platos)
        CE = np.zeros(self.num_platos)
        DE = np.zeros(self.num_platos)

        # condensador total
        BE[0] = 0.0
        CE[0] = self.hj(0) - self.Hj(1)
        DE[0] = self.F[0] * self.h_F(0) + self.Q_condensador()

        # etapa 1 hasta N - 1
        for j in self.etapas[1:-1]:
            BE[j] = self.Hj(j) - self.hj(j - 1)
            CE[j] = self.hj(j) - self.Hj(j + 1)
            DE[j] = (
                self.F[j] * self.h_F(j)
                - self.D * (self.hj(j - 1) - self.hj(j))
                - sum(self.F[k] for k in range(j + 1)) * self.hj(j)
                + sum(self.F[k] for k in range(j)) * self.hj(j - 1)
            )

        # rehervidor parcial
        BE[self.N] = self.Hj(self.N) - self.hj(self.N - 1)
        DE[self.N] = (
            self.F[self.N] * self.h_F(self.N)
            + self.Q_rehervidor()
            - self.B * (self.hj(self.N - 1) - self.hj(self.N))
            - self.F[self.N - 1] * self.hj(self.N - 1)
        )

        regulador = 1e-6
        diagonal = BE[1:] + regulador
        A = diags(
            diagonals=[diagonal, CE[1:-1]],
            offsets=[0, 1],
            shape=(self.N, self.N),
            format="csr",
        )

        self.V[1:] = linalg.spsolve(A, DE[1:])
        self.L[0] = self.R * self.D

        for j in self.etapas[1:-1]:
            self.L[j] = self.V[j + 1] - self.D + sum(self.F[k] for k in range(j + 1))

        self.L[self.N] = self.B

    def temp_burbuja(self, j):
        Lj = sum(self.l[i][j] for i in self.componentes)

        if Lj == 0:
            return self.T_ant[j]

        return punto_burbuja(
            self.T_ant[j],
            self.P_ant[j],
            [self.l[i][j] / Lj for i in self.componentes],
            [self.K_func[i].eval_SI for i in self.componentes],
        )

    def temp_al_burbuja(self):
        return punto_burbuja(
            self.T_al,
            self.P_al,
            [self.z_al[i] for i in self.componentes],
            [self.K_func[i].eval_SI for i in self.componentes],
        )

    def inicializar_caudales(self):
        self.L[: self.etp_al] = self.R * self.D
        self.L[self.etp_al : self.N] = self.R * self.D + self.F_al
        self.L[self.N] = self.B
        self.V[1:] = self.D * (self.R + 1)

    def caudales_estables(self):
        return self.error_relativo_minimo(
            self.L,
            self.L_ant,
            self.tol_caudal,
        ) and self.error_relativo_minimo(
            self.V[1:],
            self.V_ant[1:],
            self.tol_caudal,
        )

    def converger_densidades(self):
        iter = 0

        while iter < self.iter_internas:
            denL_ant = self.denL.copy()
            denV_ant = self.denV.copy()
            self.actualizar_densidades()

            if self.error_relativo_minimo(
                self.denL,
                denL_ant,
                self.tol_densidad,
            ) and self.error_relativo_minimo(
                self.denV,
                denV_ant,
                self.tol_densidad,
            ):
                break

            iter += 1

    def actualizar_densidades(self):
        R = 8314  # J/kmol/K <- gases ideales ->

        for j in self.etapas:
            xi = np.array([self.xij(i, j) for i in self.componentes])
            xi /= xi.sum()

            yi = np.array([self.yij(i, j) for i in self.componentes])
            yi /= yi.sum()

            masa_molar = np.array([self.masa_molar[i].eval() for i in self.componentes])
            denL_est = np.array([self.denL_est[i].eval() for i in self.componentes])

            self.denL[j] = (xi * masa_molar).sum()
            self.denL[j] /= (xi * masa_molar / denL_est).sum()

            self.denV[j] = self.P[j] * (yi * masa_molar).sum()
            self.denV[j] /= R * self.T[j]

    def converger_velocidades(self):
        iter = 0

        while iter < self.iter_internas:
            u_ant = self.u.copy()
            self.actualizar_velocidades()

            if self.error_relativo_minimo(
                self.u,
                u_ant,
                self.tol_velocidad,
            ):
                break

            iter += 1

    def actualizar_velocidades(self):
        k_min = 31  # <- (h_w + h_ow) vs k ->
        k_max = 0.107  # m/s <- wikipedia: souders–brown equation ->

        self.u[0] = self.u_min[0] = self.u_max[0] = 0.0

        for j in self.etapas[1:]:
            mezcla_liquida = {i: self.xij(i, j) for i in self.componentes}
            tension = tension_superficial(mezcla_liquida, self.T[j])

            masa_molar = np.array([self.masa_molar[i].eval() for i in self.componentes])
            yi = np.array([self.yij(i, j) for i in self.componentes])

            flujo_masico = self.V[j] * (yi * masa_molar).sum() / 3600

            self.u[j] = flujo_masico / (self.denV[j] * self.area_ac)

            self.u_min[j] = k_min
            self.u_min[j] *= (tension / (self.denL[j] * self.dtro_h)) ** 0.5

            self.u_max[j] = k_max
            self.u_max[j] *= ((self.denL[j] - self.denV[j]) / self.denV[j]) ** 0.5

    def converger_presiones(self):
        iter = 0

        while iter < self.iter_internas:
            self.actualizar_presiones()

            if self.presiones_estables():
                break

            iter += 1

    def actualizar_presiones(self):
        self.P[0] = self.P_ant[0] = self.P_al
        cv = 0.65  # <- manual del ingenierio químico ->
        cp = 1 / (334.76 * cv**2)  # <- dry tray pressure drop of sieve trays ->

        for j in self.etapas[1:]:
            xi = {i: self.xij(i, j) for i in self.componentes}
            tension = tension_superficial(xi, self.T[j])

            dP_vap = cp * self.u[j] ** 2 * self.denV[j]
            dP_liq = self.denL[j] * self.grav * self.alt_liq
            dP_cap = tension / self.dtro_h * (self.denL[j] / self.denV[j]) ** 0.25
            self.dP[j] = dP_vap + dP_liq + dP_cap

            Pj = self.P_al + sum(self.dP[k] for k in self.etapas[1 : j + 1])
            self.P[j] = self.P_ant[j] + self.beta * (Pj - self.P_ant[j])

    def actualizar_Ks(self):

        for i in self.componentes:
            self.K[i][:] = self.K_func[i].eval_SI(self.T[:], self.P[:])

        self.T_ant[:] = self.T[:]
        self.P_ant[:] = self.P[:]

    def masa_temperatura_energia(self):

        while True:
            self.actualizar_Ks()

            for i in self.componentes:
                self.resolver_bal_masa_comp(i)

            for j in self.etapas:
                self.T[j] = self.T_ant[j] + self.alpha * (
                    self.temp_burbuja(j) - self.T_ant[j]
                )

            if self.temperaturas_estables():
                break

        self.resolver_bal_energia()

    def espaciado_platos(self):
        k_max = 0.107
        tension = np.array(
            [
                tension_superficial(
                    {i: self.xij(i, j) for i in self.componentes}, self.T[j]
                )
                for j in self.etapas
            ]
        ).max()

        def espaciado_eq(S):
            eq = 0.193 * (self.dtro_h**2 * tension / self.denL.max()) ** 0.125
            eq *= (self.denL.max() / self.denV.max()) ** 0.1
            eq *= (S / self.alt_liq) ** 0.5

            return eq - k_max

        aprox = 1.0
        S = fsolve(espaciado_eq, aprox)[0]

        return S

    def tiempo_residencia(self):

        t_residencia = np.zeros(self.num_platos)

        for j in self.etapas:
            masa__molar = sum(
                self.xij(i, j) * self.masa_molar[i].eval() for i in self.componentes
            )
            flujo_masico = self.L[j] * masa__molar
            caudal = flujo_masico / self.denL[j]
            t_res = self.area_ac * self.alt_liq / caudal
            t_residencia[j] = t_res * 3600

        return t_residencia

    def reynolds_vapor(self):
        reynolds = np.zeros(self.num_platos)

        for j in self.etapas[1:]:
            yi = np.array([self.yij(comp, j) for comp in self.componentes])
            yi /= yi.sum()

            n = len(self.componentes)
            visc = np.zeros(n)
            msmolar = np.zeros(n)

            for idx, i in enumerate(self.componentes):
                visc[idx] = self.visc_vapor_func[i].eval(self.T[j])
                msmolar[idx] = self.masa_molar[i].eval()

            phi = np.zeros((n, n))

            for i in range(n):

                for k in range(n):
                    a = np.sqrt(visc[i] / visc[k])
                    b = (msmolar[k] / msmolar[i]) ** 0.25
                    phi[i, k] = (
                        (1.0 / np.sqrt(8.0))
                        * (1 + a * b) ** 2
                        * np.sqrt(msmolar[i] / msmolar[k])
                    )

            visc_mezcla = 0.0
            for i in range(n):
                suma_phi = np.dot(yi, phi[i, :])
                visc_mezcla += yi[i] * visc[i] / suma_phi

            reynolds[j] = self.denV[j] * self.u[j] * self.dtro_h / visc_mezcla

        return reynolds

    def run(self):
        """
        Ejecuta la secuencia completa para resolver el sistema de balances de la columna.
        """

        self.agregar_params()
        self.T_al = self.temp_al_burbuja()

        self.T[:] = self.T_al
        self.P[:] = self.P_al

        self.inicializar_caudales()
        self.masa_temperatura_energia()

        convergencia = False
        iteraciones = 0
        max_iteraciones = 150

        while not convergencia and iteraciones < max_iteraciones:
            iteraciones += 1

            self.converger_densidades()
            self.converger_velocidades()
            self.converger_presiones()
            self.masa_temperatura_energia()

            velocidades_estables = np.all(
                self.u >= self.u_min,
            ) and np.all(
                self.u <= self.u_max,
            )

            if self.caudales_estables() and velocidades_estables:
                convergencia = True

        if not convergencia:

            if velocidades_estables == False:
                pprint(("Velocidades minimas", np.around(self.u_min, 2).tolist()))
                pprint(("Velocidades de operación", np.around(self.u, 2).tolist()))
                pprint(("Velocidades maximas", np.around(self.u_max, 2).tolist()))

                raise RuntimeError(
                    "La velocidad de operación no está en el rango seguro."
                )

            raise RuntimeError(
                f"No se alcanzó la convergencia después de {iteraciones} iteraciones."
            )

        x = {}
        y = {}

        for i in self.componentes:
            x[i] = self.l[i][:] / self.L[:]
            y[i] = self.K[i][:] * x[i]

        for j in self.etapas:
            tot_x = sum(x[i][j] for i in self.componentes)
            tot_y = sum(y[i][j] for i in self.componentes)

            for i in self.componentes:
                x[i][j] /= tot_x
                y[i][j] /= tot_y

        self.S = self.espaciado_platos()

        t_residencia = self.tiempo_residencia()

        reynolds = self.reynolds_vapor()

        return {
            "Platos": self.etapas,
            "T": self.T.copy(),
            "P": self.P.copy(),
            "L": self.L.copy(),
            "V": self.V.copy(),
            "u_min": self.u_min.copy(),
            "u": self.u.copy(),
            "u_max": self.u_max.copy(),
            "denL": self.denL.copy(),
            "denV": self.denV.copy(),
            "x": x,
            "y": y,
            "S": self.S,
            "iteraciones": iteraciones,
            "Tiempos": t_residencia,
            "Reynolds": reynolds,
        }


def crear_abc(
    N: int,
    F: np.array,
    z: np.array,
    D: float,
    B: float,
    L: np.array,
    V: np.array,
    K: np.array,
):
    a = -1 * np.ones(N)  # diagonal alta
    b = np.zeros(N + 1)  # diagonal
    c = np.zeros(N)  # diagonal baja
    d = np.zeros(N + 1)

    assert (
        abs(V[0]) < 1e-8
    ), "¡La tasa de flujo de vapor fuera del condensador total no es cero!"

    # condensador total
    b[0] = 1.0 + D / L[0]
    c[0] = -V[1] * K[1] / L[1]
    d[0] = F[0] * z[0]

    # rehervidor parcial
    b[N] = 1 + V[N] * K[N] / B
    d[N] = F[N] * z[N]

    d[1:N] = F[1:N] * z[1:N]
    b[1:N] = 1 + V[1:N] * K[1:N] / L[1:N]
    c[1:N] = -V[2 : (N + 1)] * K[2 : (N + 1)] / L[2 : (N + 1)]

    return a, b, c, d


if __name__ == "__main__":
    modelo = Modelo(
        componentes=[
            "etano",
            "propano",
            "i-butano",
            "n-butano",
        ],
        z=[
            0.041,
            0.62,
            0.166,
            0.173,
        ],
        N=22,
        etp_al=13,
        F=200.0,
        D=166.0,
        R=1.307,
        T_al_apx=250.0,
        P=101325.0,
        dtro=1.7,
    )

    sol = modelo.run()

    print(
        f"""La columna cuenta con {modelo.num_platos} platos.
El espaciado entre platos es de {sol['S']:.2f} m.
Cada plato tiene un diámetro de {modelo.dtro} m y un área de {modelo.area:.2f} m^2.
La columna posee un total de {modelo.num_hs} orificios.
Cada orificio tiene un diámetro de {modelo.dtro_h} m (equivalente al {modelo.dtro_h*100/modelo.dtro:.0f}% del diámetro de la columna).
Los orificios ocupan el {modelo.area_hs*100/modelo.area_ac:.0f}% del área activa de los platos, siendo esta el {modelo.area_ac*100/modelo.area:.0f}% del área de los platos."""
    )
    print(
        f"El sistema necesitó {sol['iteraciones']} iteraciones principales para converger.",
    )
    pprint(("Temperaturas", np.around(sol["T"], 2).tolist()))
    pprint(("Presiones", np.around(sol["P"], 2).tolist()))
    pprint(("Flujos liquidos", np.around(sol["L"], 2).tolist()))
    pprint(("Flujos de vapor", np.around(sol["V"], 2).tolist()))
    pprint(("Densidades de liquidos", np.around(sol["denL"], 2).tolist()))
    pprint(("Densidades de vapor", np.around(sol["denV"], 2).tolist()))
    pprint(("Velocidades minimas", np.around(sol["u_min"], 2).tolist()))
    pprint(("Velocidades de operación", np.around(sol["u"], 2).tolist()))
    pprint(("Velocidades maximas", np.around(sol["u_max"], 2).tolist()))
    pprint(("Reynolds de vapor", np.around(sol["reynolds_vapor"], 2).tolist()))
    pprint(("Tiempos de residencia", np.around(sol["tiempo_residencia"], 2).tolist()))

    sol_x = {comp: np.around(arr, 2).tolist() for comp, arr in sol["x"].items()}
    sol_y = {comp: np.around(arr, 2).tolist() for comp, arr in sol["y"].items()}

    for comp, lista in sol_x.items():
        pprint((f"Fracción líquida {comp}", lista))

    for comp, lista in sol_y.items():
        pprint((f"Fracción vaporizada {comp}", lista))

    print(f"Q_rehervidor = {modelo.Q_rehervidor():.2f} J/h")
    print(f"Q_condensador = {modelo.Q_condensador():.2f} J/h")
