import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Classe para Contagem de Operações (Não essencial para plotagem, mas mantida para consistência) ---
class Contador:
    def __init__(self):
        self.avaliacoes = 0
        self.operacoes = 0
        self.avaliacoes_convergencia = 0
        self.operacoes_convergencia = 0

    def reset(self):
        self.avaliacoes = 0
        self.operacoes = 0
        self.avaliacoes_convergencia = 0
        self.operacoes_convergencia = 0

    def add_op(self, n=1):
        self.operacoes += n

    def registrar_ponto_convergencia(self):
        if self.avaliacoes_convergencia == 0:
            self.avaliacoes_convergencia = self.avaliacoes
            self.operacoes_convergencia = self.operacoes

contador = Contador() # Instância global

# --- Constantes Globais para F10 ---
A = 500 # [cite: 40]
B = 0.1 # [cite: 40]
C = 0.5 * np.pi # [cite: 40]

# --- Funções de Componentes para w36 (Revisadas e Consistentes com o documento) ---

# Função z: usa x_original, y_original [cite: 35]
def z_func(x_val, y_val):
    contador.add_op(2) # 2 mults
    return -x_val * np.sin(np.sqrt(np.abs(x_val))) - y_val * np.sin(np.sqrt(np.abs(y_val)))

# Função r1 (Rosenbrock Adaptada): usa x_scaled, y_scaled (x/250, y/250) [cite: 37, 38]
def r1_func(x_scaled, y_scaled):
    contador.add_op(3) # x_scaled**2, (y_s - x_s**2)**2, (1-x_s)**2
    return (y_scaled - x_scaled**2)**2 + (1 - x_scaled)**2

# Função r (Rosenbrock): usa x_scaled, y_scaled (x/250, y/250) [cite: 723]
# No Scilab original, 'r' é calculado após 'x' e 'y' serem divididos por 250.
# A expressão para 'r' é 100*(y-x.^2).^2+(1-x).^2.
# Isso é 100 * r1_func(x_scaled, y_scaled).
def r_func(x_scaled, y_scaled):
    contador.add_op(1) # 1 mult por 100
    return 100 * r1_func(x_scaled, y_scaled)

# Função Fobj: usa x_original, y_original (para derivar x1, x2 que são x_original/10, y_original/10)
# Fobj = F10 * zsh [cite: 44]
def fobj_func(x_original, y_original):
    # Conforme documento, x1 = 25 * x_scaled = 25 * (x_original/250) = x_original / 10 [cite: 41]
    # E x2 = y_original / 10[cite: 41].
    x1 = x_original / 10.0
    x2 = y_original / 10.0
    contador.add_op(2) # 2 divs (para x1, x2)

    # F10 = -A*exp(-B*sqrt((x1^2+x2^2)/2)) - exp((cos(C*x1)+cos(C*x2))/2) + exp(1) [cite: 41]
    contador.add_op(2) # x1**2, x2**2
    term_sqrt_arg = (x1**2 + x2**2) / 2.0
    contador.add_op(1) # /2.0

    contador.add_op(1) # -B * sqrt_term
    term1_f10 = -A * np.exp(-B * np.sqrt(term_sqrt_arg))
    contador.add_op(1) # *A

    contador.add_op(2) # C*x1, C*x2
    term_cos_arg = (np.cos(C * x1) + np.cos(C * x2)) / 2.0
    contador.add_op(1) # /2.0
    term2_f10 = -np.exp(term_cos_arg)

    f10 = term1_f10 + term2_f10 + np.exp(1)

    # zsh = 0.5 - ((sin(sqrt(xs^2+ys^2)))^2-0.5)/(1+0.1*(xs^2+ys^2))^2 [cite: 42]
    # 'xs' e 'ys' em zsh são interpretados como x_val/10 e y_val/10[cite: 43].
    # Isso corresponde aos nossos x1 e x2.
    arg_zsh_pow2 = x1**2 + x2**2

    numerator_sin_sqrt_arg = np.sqrt(arg_zsh_pow2)
    numerator = np.sin(numerator_sin_sqrt_arg)**2 - 0.5
    contador.add_op(1) # sin()**2

    denominator = (1 + 0.1 * arg_zsh_pow2)**2
    contador.add_op(2) # 0.1 * arg_zsh_pow2, (...)*2

    zsh = 0.5 - (numerator / denominator)
    contador.add_op(1) # /denominator

    # Fobj = F10 * zsh [cite: 44]
    contador.add_op(1) # 1 mult
    return f10 * zsh

# Função w4 (que é a w36 final)
# w4 = sqrt(r^2 + z^2) + Fobj [cite: 830, 856]
def w4_func(x_original, y_original):
    # x_scaled e y_scaled são necessários para r_func.
    x_scaled = x_original / 250.0 # [cite: 38]
    y_scaled = y_original / 250.0 # [cite: 38]
    contador.add_op(2) # 2 divs para x_scaled, y_scaled

    r_val = r_func(x_scaled, y_scaled)
    z_val = z_func(x_original, y_original)
    fobj_val = fobj_func(x_original, y_original)

    contador.add_op(2) # r**2, z**2
    return np.sqrt(r_val**2 + z_val**2) + fobj_val

# --- Função Objetivo Final w36(x,y) ---
# Esta é a função que será plotada. Ela internamente é w4_func.
# w36 = w23 + w28, que se simplifica para w36 = w4 [cite: 852, 853, 854, 855]
def w36_plot_func(x_original, y_original):
    # Para a plotagem, não estamos fazendo otimização, então o contador
    # de avaliações não é estritamente necessário aqui, mas manteremos a mesma
    # função para consistência. As operações serão contadas.
    return w4_func(x_original, y_original)

# --- Código para Plotagem ---
if __name__ == "__main__":
    # Definir o domínio para plotagem
    x_min, x_max = -500, 500 #
    y_min, y_max = -500, 500 #

    # Número de pontos em cada eixo para o meshgrid.
    # O Scilab usa 1000/5 = 200 pontos por eixo para (-500:5:500).
    # Usaremos 200 pontos para manter a consistência ou mais para um gráfico mais suave.
    num_points = 200

    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)

    # Inicializar a matriz Z para armazenar os valores da função
    Z = np.zeros(X.shape)

    print(f"Calculando valores de Z para a plotagem ({num_points}x{num_points} pontos)...")

    # Calcular Z para cada ponto (X, Y) no meshgrid
    for i in range(num_points):
        for j in range(num_points):
            # Passa os valores de X e Y para a função w36
            # O contador é resetado para cada cálculo de Z para que as operações
            # sejam contadas para CADA PONTO DA GRADE, não para o total da plotagem.
            # No entanto, para o gráfico, o mais importante é o valor Z final.
            contador.reset()
            Z[i, j] = w36_plot_func(X[i, j], Y[i, j])

    # Criar a figura e o eixo 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotar a superfície
    # Usamos cmap='viridis' para um mapa de cores que é bom para dados científicos
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Adicionar rótulos e título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('W36(X,Y)')
    ax.set_title('Gráfico da Função Objetivo W36(X,Y)')

    # Adicionar uma barra de cores para a escala de Z
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Exibir o gráfico
    plt.show()

    print("\nPlotagem concluída.")
