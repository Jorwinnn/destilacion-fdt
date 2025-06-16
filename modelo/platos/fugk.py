from modelo.punto_burbuja.estimacion import punto_burbuja
from scipy.optimize import fsolve
import numpy as np

from modelo.datos_equilibrio.depriester import DePriester

from typing import Dict, List


class Platos:
    def __init__(
        self,
        componentes: List[str],
        z: List[float],
        F: float,
        RR_min: float,
        pos_hk: int,
        f_hk: float,
        f_lk: float,
        T: float,
        P: float,
        eficiencia: float = 1.0,
    ):
        self.componentes: List[str] = componentes
        self.z = np.array(z)
        self.num_comp: int = len(z)
        self.F: float = F
        self.RR_min: float = RR_min
        self.D = np.zeros(self.num_comp)
        self.B = np.zeros(self.num_comp)
        self.T: float = T
        self.P: float = P

        self.Dx = np.zeros(self.num_comp)
        self.Dy = np.zeros(self.num_comp)
        self.Bx = np.zeros(self.num_comp)
        self.By = np.zeros(self.num_comp)
        self.KD = np.zeros(self.num_comp)
        self.KB = np.zeros(self.num_comp)
        self.K_func: Dict[str, DePriester] = {}

        self.Dvol = np.zeros(self.num_comp)
        self.Bvol = np.zeros(self.num_comp)
        self.vol_medias = np.zeros(self.num_comp)
        self.hk = pos_hk - 1
        self.lk = self.hk - 1
        self.f_hk = f_hk
        self.f_lk = f_lk
        self.eficiencia = eficiencia

    def balance_masa(self):
        self.D[: self.lk] = self.F * self.z[: self.lk]
        self.D[self.lk] = self.f_lk * self.F * self.z[self.lk]
        self.D[self.hk] = self.f_hk * self.F * self.z[self.hk]

        self.B[self.lk] = self.F * self.z[self.lk] - self.D[self.lk]
        self.B[self.hk] = self.F * self.z[self.hk] - self.D[self.hk]
        self.B[self.hk + 1 :] = self.F * self.z[self.hk + 1 :]

    def fracciones_liquidas(self):
        D = self.D.sum()
        B = self.B.sum()

        self.Dx = self.D / D
        self.Bx = self.B / B

    def K_equilibrio(self):

        for i in self.componentes:
            self.K_func[i] = DePriester(i)

    def temperatura_burbuja(self):
        self.K_equilibrio()

        TD = punto_burbuja(
            self.T,
            self.P,
            self.Dx,
            [self.K_func[i].eval_SI for i in self.componentes],
        )
        TB = punto_burbuja(
            self.T,
            self.P,
            self.Bx,
            [self.K_func[i].eval_SI for i in self.componentes],
        )

        return TD, TB

    def fracciones_vapor(self):
        TD, TB = self.temperatura_burbuja()

        for k, i in enumerate(self.componentes):
            self.KD[k] = self.K_func[i].eval_SI(TD, self.P)
            self.KB[k] = self.K_func[i].eval_SI(TB, self.P)

        self.Dy = self.KD * self.Dx
        self.By = self.KB * self.Bx

    def vol_relativas(self):
        self.Dvol = self.KD / self.KD[self.hk]
        self.Bvol = self.KB / self.KB[self.hk]
        self.vol_medias = np.sqrt(self.Dvol * self.Bvol)

    def fenske(self):
        N_min = np.log(
            self.Dx[self.lk] * self.Bx[self.hk] / (self.Dx[self.hk] * self.Bx[self.lk])
        )
        N_min /= np.log(self.vol_medias[self.lk])

        return N_min

    def underwood(self):

        def underwood_eq(theta):
            return sum(
                self.vol_medias[i] * self.z[i] / (self.vol_medias[i] - theta)
                for i in np.arange(self.num_comp)
            )

        aprox = (self.vol_medias[self.lk] + self.vol_medias[self.hk]) / 2
        theta = fsolve(underwood_eq, aprox)[0]

        R_min = (
            sum(
                self.vol_medias[i] * self.Dx[i] / (self.vol_medias[i] - theta)
                for i in np.arange(self.num_comp)
            )
            - 1.0
        )

        return R_min

    def gilliland(self, N_min, R_min):
        R = self.RR_min * R_min
        a = 1 - ((R - R_min) / (R + 1)) ** 0.5688
        a *= 0.75
        N = (N_min + a) / (1 - a)

        return N

    def kirkbride(self, N):
        NrNs = self.z[self.hk] * self.B.sum() / (self.z[self.lk] * self.D.sum())
        NrNs *= (self.Bx[self.lk] / self.Dx[self.hk]) ** 2
        NrNs **= 0.206
        etp_al = NrNs * N / (1 + NrNs)

        return etp_al

    def reflujo_real(self, R_min):
        return (self.RR_min * R_min) / self.eficiencia

    def run(self):
        self.balance_masa()
        self.fracciones_liquidas()
        self.fracciones_vapor()
        self.vol_relativas()

        N_min = self.fenske()
        R_min = self.underwood()
        N = self.gilliland(N_min, R_min)
        etp_al = self.kirkbride(N)

        from math import floor

        return {
            "N": floor(N + 0.5),
            "etp_al": floor(etp_al + 0.5),
            "D": self.D.sum(),
            "B": self.B.sum(),
            "R": self.RR_min * R_min,
            "R_min": R_min,
            "R_real": self.reflujo_real(R_min),
        }


if __name__ == "__main__":
    etapas = Platos(
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
        F=200.0,
        RR_min=1.5,
        pos_hk=4,
        f_hk=0.1,
        f_lk=0.9,
        T=250.0,
        P=101325.0,
        eficiencia=0.5,
    )

    sol = etapas.run()

    print(
        f"""Platos: {sol["N"]}, Plato alimentado: {sol["etp_al"]}
Destilado: {sol["D"]:.0f}, Fondos: {sol["B"]:.0f}
Reflujo teorico: {sol["R"]:.3f}, Reflujo mínimo: {sol["R_min"]:.3f}, Reflujo real: {sol["R_real"]:.3f}"""
    )
