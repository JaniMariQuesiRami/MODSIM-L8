import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import imageio

# Leer datos de coordenadas del archivo TSP
def read_tsp_file(filename):
    coordinates = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].isdigit():
                coordinates.append((float(parts[1]), float(parts[2])))
    return np.array(coordinates)

# Leer datos del archivo de solución óptima
def read_opt_tour_file(filename):
    tour = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().isdigit():
                tour.append(int(line.strip()) - 1)  # Convertir a índice base 0
    return tour

# Calcular la distancia total de un tour
def calculate_total_distance(tour, distances):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distances[tour[i], tour[i + 1]]
    total_distance += distances[tour[-1], tour[0]]  # Regresar al punto de inicio
    return total_distance

# Crear una población inicial de tours
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

# Función de fitness para evaluar un tour
def fitness(tour, distances):
    return 1 / calculate_total_distance(tour, distances)

# Selección de individuos para la próxima generación (Selección por Torneo)
def selection(population, distances, tournament_size=5):
    selected = []
    for _ in range(len(population) // 2):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda tour: fitness(tour, distances))
        selected.append(winner)
    return selected

# Operador de cruce para combinar dos tours (Order Crossover, OX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child

# Operador de mutación para alterar un tour (Intercambio de dos ciudades)
def mutate(tour, mutation_rate):
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

# Algoritmo genético para el TSP (con elitismo)
def genetic_algorithm_tsp(coordinates, pop_size=300, s=0.7, c=0.95, m=0.1, maxI=200000, max_stagnation=5000):
    num_cities = len(coordinates)
    distances = distance.cdist(coordinates, coordinates, 'euclidean')

    # Inicialización de la población
    population = initialize_population(pop_size, num_cities)
    best_tour = None
    best_distance = float('inf')

    # Almacenar datos para la animación
    animation_data = []

    # Variable para contar iteraciones sin mejora
    stagnation_counter = 0

    # Iteraciones del algoritmo genético
    for iteration in range(maxI):
        # Imprimir en qué iteración va cada 1000 iteraciones
        if iteration % 1000 == 0:
            print(f"Iteración: {iteration}/{maxI}")

        # Evaluación y selección
        population = selection(population, distances)

        # Cruce
        new_population = []
        while len(new_population) < pop_size * c:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            new_population.append(child)

        # Mutación
        population += [mutate(individual, m) for individual in new_population]

        # Elitismo: preservar el mejor individuo
        if best_tour is not None:
            population.append(best_tour)

        # Rellenar la población si es necesario
        while len(population) < pop_size:
            population.append(random.sample(range(num_cities), num_cities))

        # Evaluar la mejor solución de la generación
        current_best = min(population, key=lambda tour: calculate_total_distance(tour, distances))
        current_distance = calculate_total_distance(current_best, distances)

        # Verificar si hay mejora
        if current_distance < best_distance:
            best_tour, best_distance = current_best, current_distance
            stagnation_counter = 0  # Reiniciar el contador si hay mejora
        else:
            stagnation_counter += 1  # Incrementar el contador si no hay mejora

        # Almacenar datos para la animación (una muestra de iteraciones)
        if iteration % 500 == 0 or iteration == maxI - 1:
            animation_data.append((iteration, current_best.copy(), best_distance))

        # Criterio de convergencia: si no hay mejora en "max_stagnation" iteraciones, detener
        if stagnation_counter >= max_stagnation:
            print(f"Convergencia alcanzada en la iteración {iteration}.")
            break

    return best_tour, best_distance, animation_data



# Animación de los resultados y guardar como GIF
def animate_tsp(animation_data, coordinates, filename="tsp_animation.gif"):
    images = []
    fig, ax = plt.subplots()
    for iteration, tour, distance in animation_data:
        ax.clear()
        tour_coords = coordinates[tour + [tour[0]]]
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'bo-')
        ax.set_title(f'Iteración: {iteration}, Distancia: {distance:.2f}')
        plt.savefig("temp.png")
        images.append(imageio.imread("temp.png"))
    imageio.mimsave(filename, images, duration=0.5)
    plt.show()

if __name__ == "__main__":
    # Leer datos del problema
    coordinates = read_tsp_file('ch150.tsp')

    # Ejecutar el algoritmo genético
    best_tour, best_distance, animation_data = genetic_algorithm_tsp(coordinates)

    # Mostrar los resultados
    print("Tour óptimo encontrado:", best_tour)
    print("Distancia total del tour:", best_distance)

    # Animar el proceso de optimización y guardar como GIF
    animate_tsp(animation_data, coordinates)

    # Leer la solución óptima y comparar
    opt_tour = read_opt_tour_file('ch150.opt.tour')
    distances = distance.cdist(coordinates, coordinates, 'euclidean')
    opt_distance = calculate_total_distance(opt_tour, distances)

    print("\nSolución óptima indicada en el archivo:")
    print("Tour óptimo:", opt_tour)
    print("Distancia total del tour óptimo:", opt_distance)

    # Comparación de la distancia con la solución encontrada
    print("\nComparación de resultados:")
    print("Distancia del algoritmo genético:", best_distance)
    print("Distancia óptima:", opt_distance)
    if best_distance < opt_distance:
        print("El algoritmo genético encontró una mejor solución.")
    elif best_distance > opt_distance:
        print("La solución óptima es mejor que la del algoritmo genético.")
    else:
        print("El algoritmo genético encontró la solución óptima.")
