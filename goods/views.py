from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import ContactForm

import random

# Create your views here.
 

############################
def index(request):
    return render(request, 'goods/index.html')

def index2(request):
    return render(request, 'goods/index-2.html')

############################
#Dynamic algorithm for solving the fleet distribution problem
def fill_trucks(values, weights, capacities):
    num_items = len(values)
    num_trucks = len(capacities)
    dp = [[[0 for _ in range(num_trucks + 1)] for __ in range(max(capacities) + 1)] for ___ in range(num_items + 1)]
    max_values = [0] * num_trucks  # To store the maximum value for each truck

    for i in range(1, num_items + 1):
        for w in range(1, max(capacities) + 1):
            for b in range(1, num_trucks + 1):
                if weights[i-1] <= w and b-1 < num_trucks:
                    dp[i][w][b] = max(dp[i-1][w][b], dp[i-1][w-weights[i-1]][b-1] + values[i-1])
                    max_values[b-1] = max(max_values[b-1], dp[i][w][b])  # Update the maximum value for the truck
                else:
                    dp[i][w][b] = dp[i-1][w][b]

    total_max_value = sum(max_values)  # Calculate the total maximum value for all trucks
    return dp, total_max_value  # Return both the dp array and the total maximum value

def find_multi_items(dp, weights, capacities):
    num_items = len(weights)
    num_trucks = len(capacities)
    chosen_items = [[] for _ in range(num_trucks)]
    w = [cap for cap in capacities]

    for i in range(num_items, 0, -1):
        for b in range(num_trucks, 0, -1):
            if dp[i][w[b-1]][b] != dp[i-1][w[b-1]][b]:
                chosen_items[b-1].append(i-1)
                w[b-1] -= weights[i-1]

    for items in chosen_items:
        items.reverse()  # To display in the order of input

    return chosen_items

############################
#Genetic algorithm to choose the best path in terms of time
def calculate_total_distance(route, graph):
    total_distance = 0
    for i in range(len(route) - 1):
        # Check if a direct path exists between consecutive nodes
        if route[i + 1] in graph[route[i]]:
            total_distance += graph[route[i]][route[i + 1]]
        else:
            # If no direct path exists, return infinite distance
            return float('inf')
    # Check if a direct path exists from the last node to the first node
    if route[0] in graph[route[-1]]:
        total_distance += graph[route[-1]][route[0]]
    else:
        return float('inf')
    return total_distance


def create_initial_population(graph, population_size):
    population = []
    nodes = list(graph.keys())
    for _ in range(population_size):
        random_route = nodes[:]
        random.shuffle(random_route)
        if random_route[0] != 0:
            random_route.remove(0)
            random_route.insert(0, 0)
        population.append(random_route)
    return population

def fitness(route, graph):
    return 1 / calculate_total_distance(route, graph)

def select_parents(population, graph):
    fitness_scores = [fitness(route, graph) for route in population]
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parent1 = random.choices(population, probabilities)[0]
    parent2 = random.choices(population, probabilities)[0]
    return parent1, parent2

def crossover(parent1, parent2):
    child = parent1[:len(parent1)//2] + parent2[len(parent1)//2:]
    fix_duplicate_nodes(child)
    return child

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def fix_duplicate_nodes(route):
    seen = set()
    duplicates = []
    for node in route:
        if node in seen:
            duplicates.append(node)
        else:
            seen.add(node)
    for duplicate in duplicates:
        for node in set(range(len(route))) - seen:
            route[route.index(duplicate)] = node
            seen.add(node)

def genetic_algorithm(graph, population_size, generations, mutation_rate):
    population = create_initial_population(graph, population_size)
    print("Initial population created.")
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        new_population = []
        for _ in range(len(population)):
            parent1, parent2 = select_parents(population, graph)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    best_route = max(population, key=lambda route: fitness(route, graph))
    best_distance = calculate_total_distance(best_route, graph)
    print(f"The shortest path starting and ending at node 0 is: {best_route}")
    print(f"The total distance of this path is: {best_distance}")
    return best_route,best_distance

# Parameters for the genetic algorithm
population_size = 100
generations = 1000
mutation_rate = 0.01


##############################################

#The function responsible for dealing with the input data,
# passing it to the algorithms, and displaying the results to the user

def Dealing_with_dynamic_algorithm(request):
        # Collect the data entered into the form
        num_trucks= int(request.POST.get("num_trucks"))
        n= request.POST.get("number_goods")

        # Segmentation of data into a list of string numbers
        capacities= request.POST.get("truck_capacities").split(',')
        item_weights = request.POST.get("items_weight").split(',')
        item_values = request.POST.get("items_value").split(',')
        
        # Convert each item in the list to an integer
        capacities = [int(x.strip()) for x in capacities]
        item_weights = [int(x.strip()) for x in item_weights]
        item_values = [int(x.strip()) for x in item_values]
        
        # Call dynamic algorithm
        dp, total_max_value = fill_trucks(item_values, item_weights, capacities)
        chosen_items = find_multi_items(dp, item_weights, capacities)
        
        return chosen_items,total_max_value
   

def Dealing_with_genetic_algorithm(request):
    # Call the ContactForm row
    # Assign the form to POST or None
    form = ContactForm(request.POST or None)

    # If the order is of type POST
    if request.method == 'POST':

        # If the field values ​​are valid
        if form.is_valid():
            # Fetch data entered from the user and assign it to a variable
            num_addresses = int(request.POST.get("number_addresses"))
            time_input = request.POST.get("time_addresses")
            
            # An array to store the start, end and time
            times = []
            
            # Split time_input into lists to store values ​​in start, end, and time variables
            for item in time_input.split(","):
                # Split the first part to get the start
                parts1 = item.split("->")
                start = int(parts1[0])
                # Divide the second part to get the end and time
                parts2 = parts1[1].split("=")
                end = int(parts2[0])
                time = int(parts2[1])
                times.append((start, end, time))

            # Create a dictionary to store the time between every two points
            graph = {i: {} for i in range(num_addresses)}
            for start, end, time in times:
                graph[start][end] = time

            # Call the function handling the dynamic algorithm
            chosen_items, total_max_value = Dealing_with_dynamic_algorithm(request)
            # Call the genetic algorithm function
            shortest_route, shortest_distance = genetic_algorithm(graph, population_size, generations, mutation_rate)
            
           # Pass the result to the shortest_route page
            return render(request, 'goods/show_results.html', {
                'chosen_items': chosen_items,
                'total_max_value': total_max_value,
                'shortest_route': shortest_route,
                'shortest_distance': shortest_distance
            })
   
   # Display the form page if the request is not of POST type
    return render(request, 'goods/index-3.html' ,{'form': form})