import random
import numpy as np
from typing import List, Dict, Callable

def objective_function(x: float, y: float) -> float:
    z = -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

    x_scaled, y_scaled = x / 250.0, y / 250.0

    r = 100 * (y_scaled - x_scaled**2)**2 + (1 - x_scaled)**2

    a, b, c = 500.0, 0.1, 0.5 * np.pi
    x1, y1 = 25 * x_scaled, 25 * y_scaled

    term1_f10 = -a * np.exp(-b * np.sqrt((x1**2 + y1**2) / 2))
    term2_f10 = -np.exp((np.cos(c * x1) + np.cos(c * y1)) / 2)
    f10 = term1_f10 + term2_f10 + np.exp(1)

    xs2_ys2 = x1**2 + y1**2
    zsh = 0.5 - ((np.sin(np.sqrt(xs2_ys2)))**2 - 0.5) / (1 + 0.1 * xs2_ys2**2 + 1e-9)
    f_obj = f10 * zsh

    final_value = np.sqrt(r**2 + z**2) + f_obj
    return final_value

# CLASSE DO ALGORITMO GENÉTICO
class GeneticAlgorithm:
    def __init__(self,
                 objective_func: Callable,
                 bounds: List[float],
                 pop_size: int,
                 generations: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 bits_per_var: int):

        self.objective_func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.bits_per_var = bits_per_var
        self.chromosome_length = bits_per_var * 2
        self.function_evaluations = 0

    def _decode_chromosome(self, chromosome: List[int]) -> tuple[float, float]:
        """Decodifica um cromossomo binário para os valores (x, y)."""
        # Traduz um cromossomo em bits para int
        x_bits = chromosome[:self.bits_per_var]
        y_bits = chromosome[self.bits_per_var:]
        x_int = int("".join(map(str, x_bits)), 2)
        y_int = int("".join(map(str, y_bits)), 2)
        max_int_val = 2**self.bits_per_var - 1

        x = self.bounds[0] + (x_int / max_int_val) * (self.bounds[1] - self.bounds[0])
        y = self.bounds[0] + (y_int / max_int_val) * (self.bounds[1] - self.bounds[0])
        return x, y

    def _calculate_fitness(self, chromosome: List[int]) -> float:
        """Calcula a aptidão de um cromossomo, contando a avaliação."""
        x, y = self._decode_chromosome(chromosome)
        value = self.objective_func(x, y)
        self.function_evaluations += 1
        return -value # para que "menor w36" se torne "maior fitness"
        """ O algoritmo precisa ser instruído a ter valores menores
        De uma aptidão negativa se torna uma muito positiva, podendo até ser ótima """

    def _selection(self, population_with_fitness: List[tuple]) -> List[int]:
        """Seleção por Roleta."""
        # A probabilidade de um indivíduo ser escolhido é proporcional à sua aptidão
        fitnesses = [fit for _, fit in population_with_fitness]
        # o método da roleta não funciona com valores negativos
        min_fit = min(fitnesses) if fitnesses else 0
        adjusted_fitnesses = [(fit - min_fit) + 1e-6 for fit in fitnesses] # colocado para não ter confusão nos cálculos por um número ser 0
        # Normaliza (fit - min_fit) para garantir que todas as aptidões se tornem positivas antes da seleção
        total_fit = sum(adjusted_fitnesses) # soma das aptidões, que é o tamanho total da roleta

        pick = random.uniform(0, total_fit) # "girar a roleta"
        current_sum = 0
        for i, adj_fit in enumerate(adjusted_fitnesses):
            current_sum += adj_fit
            if current_sum > pick:
                return population_with_fitness[i][0]
        return population_with_fitness[-1][0]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> tuple[List[int], List[int]]:
        """Crossover de ponto único."""
        if random.random() < self.crossover_rate: # Se o crossover_rate for 80%, o cruzamento ocorre. Caso contrário, os pais são passados para a próxima geração sem alterações
            point = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2 # Se for satisfeita (o número aleatório for maior ou igual à taxa de crossover), o cruzamento não é realizado, e a função simplesmente retorna os dois pais originais sem nenhuma modificação

    def _mutation(self, chromosome: List[int]) -> List[int]:
        """Mutação de bit."""
        return [(1 - bit) if random.random() < self.mutation_rate else bit for bit in chromosome]
        # A mutação é um processo feito aleatoriamente

    def run(self) -> Dict:
        """Executa o fluxo principal do Algoritmo Genético."""
        population = [[random.randint(0, 1) for _ in range(self.chromosome_length)] for _ in range(self.pop_size)]
        best_solution_overall = {'value': float('inf'), 'chromosome': None}
        self.function_evaluations = 0

        print("Iniciando a otimização por Algoritmo Genético")
        for gen in range(self.generations):
            population_with_fitness = [(chromo, self._calculate_fitness(chromo)) for chromo in population]

            for individual, fitness in population_with_fitness:
                current_value = -fitness
                if current_value < best_solution_overall['value']:
                    best_solution_overall['value'] = current_value
                    best_solution_overall['chromosome'] = individual

            next_population = [best_solution_overall['chromosome']]

            while len(next_population) < self.pop_size:
                parent1 = self._selection(population_with_fitness)
                parent2 = self._selection(population_with_fitness)
                child1, child2 = self._crossover(parent1, parent2)

                next_population.append(self._mutation(child1))
                if len(next_population) < self.pop_size:
                    next_population.append(self._mutation(child2))

            population = next_population

            if (gen + 1) % 20 == 0:
                print(f"Geração {gen+1:03d}: Melhor Valor W36 = {best_solution_overall['value']:.4f}")

        final_x, final_y = self._decode_chromosome(best_solution_overall['chromosome'])
        best_solution_overall['x'] = final_x
        best_solution_overall['y'] = final_y
        best_solution_overall['total_evaluations'] = self.function_evaluations

        return best_solution_overall

if __name__ == "__main__":
    ga = GeneticAlgorithm(
        objective_func=objective_function,
        bounds=[-500, 500],
        pop_size=40,
        generations=200,
        crossover_rate=0.8,
        mutation_rate=0.01,
        bits_per_var=24
    )

    final_solution = ga.run()

    print("\n--- Resultado Final do Algoritmo Genético ---")
    print(f"Melhor valor mínimo encontrado para W36(x,y): {final_solution['value']:.12f}")
    print(f"Obtido com os valores de x e y:")
    print(f"x = {final_solution['x']:.16f}")
    print(f"y = {final_solution['y']:.16f}")
    print(f"Total de Avaliações da Função Objetivo: {final_solution['total_evaluations']}")
