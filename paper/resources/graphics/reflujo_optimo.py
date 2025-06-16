import numpy as np
import matplotlib.pyplot as plt

color1 = "red"
color2 = "green"
color3 = "blue"
color = "black"


# Función de transición suave (sigmoide)
def smooth_transition(x, center, width):
    return 1 / (1 + np.exp(-10 * (x - center) / width))


# Parámetros de transición
transition_center = 3.77
transition_width = 0.15

# Crear arrays unificados para las curvas principales
x_combined = np.linspace(2, 11, 500)

# Función combinada para Costos Fijos (rojo)
y_red = np.zeros_like(x_combined)
for i, x in enumerate(x_combined):
    if x <= transition_center - transition_width:
        y_red[i] = 0.15 * (x - 7.6) ** 2
    elif x >= transition_center + transition_width:
        y_red[i] = np.exp(0.04 * (x - 2) ** 2) + 1
    else:
        blend = smooth_transition(x, transition_center, transition_width)
        y_left = 0.15 * (x - 7.6) ** 2
        y_right = np.exp(0.04 * (x - 2) ** 2) + 1
        y_red[i] = (1 - blend) * y_left + blend * y_right

# Función combinada para Costo Total (azul)
y_blue = np.zeros_like(x_combined)
for i, x in enumerate(x_combined):
    if x <= transition_center - transition_width:
        y_blue[i] = 0.15 * (x - 4) ** 2 + 6.5
    elif x >= transition_center + transition_width:
        y_blue[i] = np.exp(0.05 * (x - 1) ** 2) + 5
    else:
        blend = smooth_transition(x, transition_center, transition_width)
        y_left = 0.15 * (x - 4) ** 2 + 6.5
        y_right = np.exp(0.05 * (x - 1) ** 2) + 5
        y_blue[i] = (1 - blend) * y_left + blend * y_right

# Elementos restantes del gráfico
x3 = np.linspace(2, 11, 50)
y3 = 0.15 * x3**2

x4 = np.ones(50) * 2
y4 = np.linspace(0, 10, 50)

x5 = np.ones(50) * 3.77
y5 = np.linspace(2.2, 6.5, 50)

plt.figure(figsize=(5, 5))

# Curvas principales
plt.plot(x_combined, y_red, color=color1, label="Costos fijos")
plt.plot(x3, y3, color=color2, label="Costos operativos")
plt.plot(x_combined, y_blue, color=color3, label="Costo total")

# Elementos auxiliares
plt.plot(x5, y5, "--", color=color, label="Reflujo óptimo")
plt.plot(x4, y4, color=color)

plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])

plt.grid(True)
plt.xlabel("Relación de reflujo")
plt.ylabel("Costo anual [dolares/año]")
plt.xlim(0, 11)
plt.ylim(0, 10)
plt.tight_layout()
plt.legend()
plt.savefig("paper/resources/images/reflujo_optimo.pdf", format="pdf")
# plt.show()
