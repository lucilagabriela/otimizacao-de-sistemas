import random
import numpy as np

# --- 1. DEFINIÇÃO DA FUNÇÃO OBJETIVO ---
def objective_function(x, y):
    # Componente z (coordenadas originais)
    z = -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

    # Escalonamento para o componente r
    x_scaled = x / 250.0
    y_scaled = y / 250.0

    # Componente r (Rosenbrock)
    r = 100 * (y_scaled - x_scaled**2)**2 + (1 - x_scaled)**2

    # Componente Fobj (Ackley + Schaffer)
    a, b, c = 500.0, 0.1, 0.5 * np.pi
    x1, y1 = 25 * x_scaled, 25 * y_scaled

    term1_f10 = -a * np.exp(-b * np.sqrt((x1**2 + y1**2) / 2))
    term2_f10 = -np.exp((np.cos(c * x1) + np.cos(c * y1)) / 2)
    f10 = term1_f10 + term2_f10 + np.exp(1)

    xs2_ys2 = x1**2 + y1**2
    zsh = 0.5 - ((np.sin(np.sqrt(xs2_ys2)))**2 - 0.5) / (1 + 0.1 * xs2_ys2)**2
    f_obj = f10 * zsh

    final_value = np.sqrt(r**2 + z**2) + f_obj
    return final_value

# Estes são os parâmetros que você deve ajustar para a sua função
N_PARTICLES = 50          # Número de partículas no enxame
N_ITERATIONS = 200        # Número de iterações
VAR_BOUNDS = [-500, 500]  # Limites do espaço de busca
DIMENSIONS = 2            # Número de variáveis (x, y)

# Coeficientes do PSO
w = 0.5   # Peso da inércia
c1 = 1.5  # Coeficiente cognitivo (pessoal)
c2 = 1.5  # Coeficiente social (global)

def particle_swarm_optimization():
    # --- Inicialização do Enxame ---
    swarm = [] # Enxame
    for _ in range(N_PARTICLES):
        position = np.random.uniform(low=VAR_BOUNDS[0], high=VAR_BOUNDS[1], size=DIMENSIONS)
        velocity = np.zeros(DIMENSIONS)
        pbest_position = position.copy()
        pbest_value = objective_function(position[0], position[1])
        swarm.append({'position': position, 'velocity': velocity, 'pbest_position': pbest_position, 'pbest_value': pbest_value})

    # Inicialização do gbest (melhor global)
    gbest_value = float('inf')
    gbest_position = np.zeros(DIMENSIONS)

    for particle in swarm:
        if particle['pbest_value'] < gbest_value:
            gbest_value = particle['pbest_value']
            gbest_position = particle['pbest_position'].copy()

    print("Iniciando a otimização por PSO")

    # Loop Principal de Otimização
    for i in range(N_ITERATIONS):
        for particle in swarm:
            # Atualização da Velocidade
            r1 = np.random.rand(DIMENSIONS)
            r2 = np.random.rand(DIMENSIONS)

            inertia_term = w * particle['velocity']
            cognitive_term = c1 * r1 * (particle['pbest_position'] - particle['position'])
            social_term = c2 * r2 * (gbest_position - particle['position'])

            new_velocity = inertia_term + cognitive_term + social_term
            particle['velocity'] = new_velocity

            # Atualização da Posição
            new_position = particle['position'] + new_velocity

            # Garante que a partícula não saia dos limites do espaço de busca
            particle['position'] = np.clip(new_position, VAR_BOUNDS[0], VAR_BOUNDS[1])

            # Avaliação e Atualização das Memórias
            current_value = objective_function(particle['position'][0], particle['position'][1])

            # Atualiza o pbest
            if current_value < particle['pbest_value']:
                particle['pbest_value'] = current_value
                particle['pbest_position'] = particle['position'].copy()

            # Atualiza o gbest
            if current_value < gbest_value:
                gbest_value = current_value
                gbest_position = particle['position'].copy()

        if (i + 1) % 20 == 0: # Imprime o progresso a cada 20 iterações
            print(f"Iteração {i+1:03d}: Melhor Valor W36 = {gbest_value:.4f}")

    return gbest_position, gbest_value

final_position, final_value = particle_swarm_optimization()
print("\n--- Resultado Final do PSO ---")
print(f"Melhor valor mínimo encontrado para W36(x,y): {final_value:.12f}")
print(f"Obtido na posição (x, y):")
print(f"x = {final_position[0]:.16f}")
print(f"y = {final_position[1]:.16f}")
