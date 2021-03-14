import Graph_Class as gc 


graph = gc.Graph(filename="Data/adj_matrix_7.txt", check=True)     # Leggo matrice di adiacenza da file e calcolo numero di nodi, numero di archi e grado massimo. controllo che sia tutto a posto

print("Matrice di Adiacenza del Grafo:")
print(graph.adj_matrix)
print("Numero di nodi: ", graph.n_nodes)
print("Numero di archi: ", graph.m_edges)
print("Grado massimo: ", graph.max_degree)

colors_used = graph.Greedy_Coloring("Data/available_colors.txt")

print("Ordine dei colori assegnati: ", graph.vertex_color)
print("L'algoritmo Greedy ha utilizzato ", colors_used, " colori")
graph.Print()






