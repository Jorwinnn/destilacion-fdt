import flet as ft
import matplotlib.pyplot as plt

from flet.matplotlib_chart import MatplotlibChart
from matplotlib import use as matplotlib_use

matplotlib_use("svg")

from modelo.platos.fugk import Platos
from modelo.solucion_numerica.main import Modelo


class UI:
    def mi_textfield(
        self,
        tvalue=None,
        tfwidth=110,
        tfvalue=None,
        tfsuffix=None,
        tficon=None,
        tfmaxlength=5,
        tfdisabled=True,
    ):
        textfield = ft.TextField(
            width=tfwidth,
            value=tfvalue,
            text_size=15,
            color=ft.Colors.WHITE,
            filled=True,
            fill_color=self.color_controles,
            border_radius=20,
            border_color=ft.Colors.WHITE,
            text_align=ft.TextAlign.CENTER,
            suffix=ft.Text(
                value=tfsuffix,
                size=15,
                color=ft.Colors.WHITE,
            ),
            max_length=tfmaxlength,
            disabled=tfdisabled,
        )

        if tficon:
            textfield.suffix_icon = ft.Icon(
                name=tficon,
                color=ft.Colors.WHITE,
                size=20,
            )

        return ft.Container(
            expand=True,
            alignment=ft.alignment.center,
            content=ft.Row(
                expand=True,
                controls=[
                    ft.Text(
                        expand=True,
                        value=tvalue,
                        size=15,
                        color=ft.Colors.WHITE,
                        text_align=ft.TextAlign.LEFT,
                    ),
                    textfield,
                ],
            ),
        )

    def __init__(self):
        self.color_controles = "#655cb6"

        self.fracciones = ft.Column(
            expand=5,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            spacing=0,
            controls=[
                ft.Text(
                    value="Fracciones\nmolares",
                    size=18,
                    color=ft.Colors.WHITE,
                    text_align=ft.TextAlign.CENTER,
                ),
                self.mi_textfield(
                    tvalue="Etano:",
                    tfvalue="0.041",
                    tficon=ft.Icons.BALANCE,
                ),
                self.mi_textfield(
                    tvalue="Propano:",
                    tfvalue="0.62",
                    tficon=ft.Icons.BALANCE,
                ),
                self.mi_textfield(
                    tvalue="i-Butano:",
                    tfvalue="0.166",
                    tficon=ft.Icons.BALANCE,
                ),
                self.mi_textfield(
                    tvalue="n-Butano:",
                    tfvalue="0.173",
                    tficon=ft.Icons.BALANCE,
                ),
            ],
        )

        self.diseño = ft.Column(
            expand=3,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            spacing=0,
            controls=[
                ft.Text(
                    value="Parámetros\nde diseño",
                    size=18,
                    color=ft.Colors.WHITE,
                    text_align=ft.TextAlign.CENTER,
                ),
                self.mi_textfield(
                    tfwidth=100,
                    tvalue="Diametro:",
                    tfvalue="1.7",
                    tfsuffix="m",
                    tficon=ft.Icons.SETTINGS,
                    tfmaxlength=3,
                ),
                self.mi_textfield(
                    tfwidth=100,
                    tvalue="Eficiencia:",
                    tfvalue="0.5",
                    tficon=ft.Icons.SETTINGS,
                ),
            ],
        )

        self.operativos = ft.Column(
            expand=1,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            spacing=0,
            controls=[
                ft.Text(
                    value="Parámetros\noperativos",
                    size=18,
                    color=ft.Colors.WHITE,
                    text_align=ft.TextAlign.CENTER,
                ),
                self.mi_textfield(
                    tvalue="Entrada:",
                    tfvalue="200",
                    tfwidth=120,
                    tfsuffix=" kmol/h",
                ),
                self.mi_textfield(
                    tvalue="R\nRmin:",
                    tfwidth=70,
                    tfvalue="1.5",
                ),
                self.mi_textfield(
                    tvalue="Fracción\nLK:",
                    tfwidth=70,
                    tfvalue="0.1",
                ),
                self.mi_textfield(
                    tvalue="Fracción\nHK:",
                    tfwidth=70,
                    tfvalue="0.9",
                ),
                self.mi_textfield(
                    tvalue="HK:",
                    tfwidth=110,
                    tfvalue="n-Butano",
                ),
                self.mi_textfield(
                    tvalue="Presión\nTope:",
                    tfwidth=110,
                    tfvalue="101325",
                    tfsuffix=" Pa",
                ),
            ],
        )

        datos = self.datos()
        self.eje_x = ft.Dropdown(
            width=135,
            filled=True,
            fill_color=self.color_controles,
            bgcolor=self.color_controles,
            border_radius=20,
            text_size=14,
            label="Eje x",
            label_style=ft.TextStyle(
                size=15,
                color=ft.Colors.WHITE,
            ),
            options=[ft.dropdown.Option(dato) for dato in datos],
        )

        self.eje_y = ft.Dropdown(
            width=135,
            filled=True,
            fill_color=self.color_controles,
            bgcolor=self.color_controles,
            border_radius=20,
            text_size=14,
            label="Eje y",
            label_style=ft.TextStyle(
                size=15,
                color=ft.Colors.WHITE,
            ),
            options=[ft.dropdown.Option(dato) for dato in datos],
        )

        self.simulacion = ft.Row(
            expand=1,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=30,
            controls=[
                ft.Text(
                    value="Simulación\nSI",
                    size=18,
                    color=ft.Colors.WHITE,
                    text_align=ft.TextAlign.CENTER,
                ),
                self.eje_x,
                self.eje_y,
                ft.Container(
                    alignment=ft.alignment.center,
                    content=ft.TextButton(
                        width=120,
                        height=40,
                        text="Graficar",
                        icon=ft.Icons.PLAY_ARROW,
                        style=ft.ButtonStyle(
                            text_style=ft.TextStyle(size=15, weight=ft.FontWeight.BOLD),
                            color=ft.Colors.WHITE,
                            bgcolor={
                                ft.ControlState.DEFAULT: self.color_controles,
                                ft.ControlState.PRESSED: ft.Colors.LIGHT_BLUE_ACCENT_700,
                            },
                        ),
                        on_click=lambda _: self.actualizacion(),
                    ),
                ),
            ],
        )

        self.resultados = ft.Column(
            expand=1,
            horizontal_alignment=ft.CrossAxisAlignment.START,
            spacing=0,
            controls=[
                ft.Text(
                    value="Resultados\nnúmericos",
                    size=18,
                    color=ft.Colors.WHITE,
                    text_align=ft.TextAlign.CENTER,
                ),
                self.mi_textfield(
                    tvalue="Platos:",
                    tfwidth=90,
                ),
                self.mi_textfield(
                    tvalue="Número\nentrada:",
                    tfwidth=90,
                ),
                self.mi_textfield(
                    tvalue="Reflujo\nreal:",
                    tfwidth=90,
                ),
                self.mi_textfield(
                    tvalue="Espaciado:",
                    tfwidth=90,
                    tfsuffix=" m",
                ),
                self.mi_textfield(
                    tvalue="Diámetro\nhuecos:",
                    tfwidth=90,
                    tfsuffix=" m",
                ),
                self.mi_textfield(
                    tvalue="Cantidad\nhuecos:",
                    tfwidth=90,
                ),
            ],
        )

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.grafico_actual: MatplotlibChart = MatplotlibChart(
            expand=9,
            figure=self.fig,
        )

        self.cuerpo = ft.Container(
            expand=True,
            alignment=ft.alignment.center,
            padding=ft.padding.all(value=30),
            gradient=ft.LinearGradient(
                begin=ft.alignment.top_left,
                end=ft.alignment.bottom_right,
                colors=[
                    "#0f0c29",
                    "#434393",
                    "#0f0c29",
                ],
            ),
            content=ft.Row(
                expand=True,
                spacing=30,
                controls=[
                    ft.Column(
                        expand=1,
                        spacing=0,
                        controls=[
                            self.fracciones,
                            self.diseño,
                        ],
                    ),
                    self.operativos,
                    ft.Column(
                        expand=3,
                        spacing=0,
                        controls=[
                            self.simulacion,
                            self.grafico_actual,
                        ],
                    ),
                    self.resultados,
                ],
            ),
        )

    def datos(self):
        return [
            "Platos",
            "T",
            "P",
            "L",
            "V",
            "x",
            "y",
            "u_min",
            "u",
            "u_max",
            "denL",
            "denV",
            "Tiempos",
            "Reynolds",
        ]

    def actualizacion(self):
        if self.eje_x.value == None or self.eje_y.value == None:
            return

        self.actualizar_graficas()
        self.actualizar_resultados()

    def actualizar_graficas(self):
        etapas = Platos(
            componentes=[
                "etano",
                "propano",
                "i-butano",
                "n-butano",
            ],
            z=[
                float(self.fracciones.controls[1].content.controls[1].value),
                float(self.fracciones.controls[2].content.controls[1].value),
                float(self.fracciones.controls[3].content.controls[1].value),
                float(self.fracciones.controls[4].content.controls[1].value),
            ],
            F=float(self.operativos.controls[1].content.controls[1].value),
            RR_min=float(self.operativos.controls[2].content.controls[1].value),
            pos_hk=4,
            f_hk=float(self.operativos.controls[3].content.controls[1].value),
            f_lk=float(self.operativos.controls[4].content.controls[1].value),
            T=250.0,
            P=float(self.operativos.controls[6].content.controls[1].value),
            eficiencia=float(self.diseño.controls[2].content.controls[1].value),
        )
        sol = etapas.run()

        self.modelo = Modelo(
            componentes=[
                "etano",
                "propano",
                "i-butano",
                "n-butano",
            ],
            z=[
                float(self.fracciones.controls[1].content.controls[1].value),
                float(self.fracciones.controls[2].content.controls[1].value),
                float(self.fracciones.controls[3].content.controls[1].value),
                float(self.fracciones.controls[4].content.controls[1].value),
            ],
            N=sol["N"] + 1,
            etp_al=sol["etp_al"] + 1,
            F=float(self.operativos.controls[1].content.controls[1].value),
            D=sol["D"],
            R=sol["R_real"],
            T_al_apx=250.0,
            P=float(self.operativos.controls[6].content.controls[1].value),
            dtro=float(self.diseño.controls[1].content.controls[1].value),
        )
        datos = self.modelo.run()

        self.ax.clear()

        comps = self.modelo.componentes
        eje_x = self.eje_x.value
        eje_y = self.eje_y.value
        colors = [
            "#E63946",
            "#2A9D8F",
            "#F4A261",
            "#264653",
        ]

        # Caso especial: fracciones molares vs plato -> barras apiladas verticales
        if eje_y in ["x", "y"] and eje_x not in ["x", "y"]:
            fracs_dict = datos[eje_y]
            num_platos = self.modelo.num_platos
            etapas_idx = list(range(1, num_platos + 1))
            width = 0.8
            import numpy as np

            for etapa in etapas_idx:
                fracs = [fracs_dict[comp][etapa - 1] for comp in comps]
                total = sum(fracs)

                if abs(total - 1.0) > 1e-6:
                    fracs = [f / total for f in fracs]

                bottoms = np.concatenate(([0], np.cumsum(fracs)))[:-1]

                for idx, (h, bot) in enumerate(zip(fracs, bottoms)):
                    self.ax.bar(etapa, h, bottom=bot, width=width, color=colors[idx])

            self.ax.set_xlim(0.5, num_platos + 0.5)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlabel("Plato", fontsize=14)
            self.ax.set_ylabel("Fracción molar", fontsize=14)

            from matplotlib.patches import Patch

            legend_handles = [
                Patch(color=colors[i], label=comps[i]) for i in range(len(comps))
            ]

            self.ax.legend(
                handles=legend_handles,
                title="Componentes",
                title_fontsize=12,
                fontsize=11,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=len(comps),
                frameon=False,
            )

            self.fig.tight_layout()
            self.grafico_actual.update()
            return

        x_vec = eje_x in ["x", "y"]
        y_vec = eje_y in ["x", "y"]

        if x_vec or y_vec:
            for idx, comp in enumerate(comps):
                if x_vec and y_vec:
                    xs = datos[eje_x][comp]
                    ys = datos[eje_y][comp]
                elif x_vec:
                    xs = datos[eje_x][comp]
                    ys = datos[eje_y]
                else:
                    xs = datos[eje_x]
                    ys = datos[eje_y][comp]
                self.ax.scatter(xs, ys, color=colors[idx], label=comp)
        else:
            xs = datos[eje_x]
            ys = datos[eje_y]
            self.ax.scatter(xs, ys, color="purple", label=f"{eje_x} vs {eje_y}")

        self.ax.set_title(f"{eje_x} vs {eje_y}", fontsize=16)
        self.ax.set_xlabel(eje_x, fontsize=14)
        self.ax.set_ylabel(eje_y, fontsize=14)
        self.ax.tick_params(axis="both", which="major", labelsize=12)
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if x_vec or y_vec:
            legend = self.ax.legend(
                title="Componente",
                title_fontsize=12,
                fontsize=11,
                loc="upper left",
                frameon=True,
            )
            legend.get_frame().set_alpha(0.9)

        self.fig.tight_layout()
        self.grafico_actual.update()

    def actualizar_resultados(self):
        self.resultados.controls[1].content.controls[1].value = self.modelo.num_platos
        self.resultados.controls[2].content.controls[1].value = self.modelo.etp_al
        self.resultados.controls[3].content.controls[1].value = f"{self.modelo.R:.3f}"
        self.resultados.controls[4].content.controls[1].value = f"{self.modelo.S:.1f}"
        self.resultados.controls[5].content.controls[1].value = self.modelo.dtro_h
        self.resultados.controls[6].content.controls[1].value = self.modelo.num_hs

        self.resultados.update()


def main(page: ft.Page):
    ui = UI().cuerpo

    page.padding = 0
    page.title = "Destilación multicomponente (Platos perforados)"
    page.fonts = {"fira_code": "fonts/FiraCode.ttf"}
    page.theme = ft.Theme(font_family="fira_code")
    page.window.maximized = True
    # page.window.center()

    page.add(ui)


if __name__ == "__main__":
    ft.app(target=main)
