import numpy as np 
import string
import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx 

# TODO: verificare connessione con DFS (calcolo già albero)
# TODO: in caso di grafo non connesso dividere in sottografi connessi ed eseguire greedy separatamente
# TODO: implementare ordinamento nel caso di grafi regolari in modo da far valere teorema di Brooks

''' 
    ***** CLASSE GRAPH ******
    Rappresentazione scelta: matrici di adiacenza

    Variabili:      adj_matrix          matrice di adiacenza                                                    array numpy nxn
                    n_nodes             numero di nodi       
                    m_edges             numero di archi      
                    max_degree          grado massimo tra tutti i vertici
                    span_tree           spanning tree                                                           variabile di tipo Graph
                    vertex_order        array contenente l'ordine degli indici (e dunque dei vertici)           array numpy n
                    vertex_color        array contenente i colori di ciascun vertice (ordinamento originale)    array numpy n

    Metodi:         costruttore, Set_Up, Check              crea matrice di adiacenza da file o da matrice preesistente, inizializza altre variabili, controlla condizioni sulla matrice
                    Depth_First_Search, Visita              creazione di uno spanning tree
                    Vertex_Ordering                         ordinamento degli indici tramite Dijkstra e Merge_Sort
                    Greedy_Coloring, Color_Already_Used     coloro in base a ordinamento precedente 
                    Dijkstra                                trovo vettore di distanze da una sorgente a tutti gli altri vertici
                    (Metodi legati al Check)
                    (Metodi per calcolare grado, cammino...)

    Funzioni:       Merge_Sort, Merge    Ordinamento di due vettori (confronti fatti su uno dei due) in ordine DECRESCENTE!

'''

class Graph:

    # *********** METODI DI COSTRUZIONE *************

    def __init__(self, filename="", matrix=None, check=False):
        if filename:
            self.adj_matrix = np.loadtxt(filename, dtype='i', delimiter=' ')          # Leggo la matrice di adiacenza contenuta nel file di input
        elif matrix is not None:
            self.adj_matrix = np.copy(matrix)                                         # Passo direttamente la matrice al costruttore
        else:
            self.adj_matrix=None
        
        self.Set_Up(check)

    def Set_Up(self, check):                                                                  # Da chiamare quando associo una nuova matrice di adiacenza
        if self.adj_matrix is not None:
            self.n_nodes = len(self.adj_matrix)                                        # Numero dei nodi pari al numero di righe della matrice (se non corrisponde con numero colonne programma terminerà con Check)
            self.max_degree = self.Max_Degree()                                        # Grado massimo del grafo
            if check:                                                                  # Se devo controllare caratteristiche della matrice di adiacenza lo faccio prima di calcolare il numero di archi ma dopo aver calcolato il numero di nodi e il grado massimo!
                self.Check()
            self.m_edges = self.Calculate_Edges()                                      # Numero degli archi (trovo elementi della matrice pari ad 1)

        else:
            self.n_nodes = None
            self.m_edges = None
            self.max_degree = None
        
        self.span_tree = None                                                     # Conterrà lo spanning tree (a sua volta un grafo)
        self.vertex_order = None                                                  # Vettore che contiene l'ordine dei vertici da colorare
        self.vertex_color = None                                                  # Vettore che racchiude il colore associato ad ogni vertice


    def Check(self):
        if not self.Symmetry_Matrix():                                            # Se dimensioni sono giuste controllo che matrice sia simmetrica (ed eventualmente correggo)
            print("Le dimensioni della matrice di adiacenza non sono corrette. Il programma non può continuare")
            sys.exit()
        
        self.Loop_Matrix()                                                        # Se ci sono dei Loop correggo

        if self.Regular_Matrix():                                                 # Se matrice regolare blocco programma (si potrebbe comunque colorare con Greedy ma teorema di Brooks non verrebbe più rispettato con l'attuale ordinamento!)
            print("Il grafo è regolare. Il programma non può continuare.")
            sys.exit()

        if not self.Connected_Matrix():                                           # Se grafo non connesso dovrei dividere in due sottografi ed eseguire Greedy separatamente...
            print("Il grafo non è connesso. Il programma non può continuare")
            sys.exit()
 



    # ************** SPANNING TREE **************

    def Depth_First_Search(self):                               # Trovo uno spanning Tree con metodo della ricorsione
        vertex_array = np.full(self.n_nodes, "i")               # Inizialmente tutti i nodi isolati
        adj_spanning=np.zeros((self.n_nodes, self.n_nodes))     # Matrice di supporto dello Spanning Tree nulla
        self.Visita(0, vertex_array, adj_spanning)              # Incomincio dal primo nodo
        self.span_tree = Graph(matrix=adj_spanning)         # Lo spanning Tree è un ulteriore grafo costruito passando la matrice di adiacenza al costruttore!!!
        #print(self.span_tree.adj_matrix)

    def Visita(self, iterator, vertex_array, adj_spanning):
        vertex_array[iterator]="a"                              # Dichiaro l'elemento iniziale aperto
        for i in range(self.n_nodes):
            if self.adj_matrix[iterator][i] == 1 and vertex_array[i]=="i":           # Se trovo elemento adiacente a iter inesplorato aggiungo arco allo spanning tree dell'albero
                adj_spanning[i][iterator] = 1
                adj_spanning[iterator][i] = 1
                self.Visita(i,vertex_array,adj_spanning)
        vertex_array[iterator]="c"                              # Chiudo il vertice


    # ************** ORDINAMENTO DEI VERTICI PER COLORAMENTO GREEDY ****************

    def Vertex_Ordering(self):
        if self.span_tree is None:
            self.Depth_First_Search()
        
        self.vertex_order = np.arange(self.n_nodes, dtype=int)        # Ordine iniziale: indici da 0 a n_nodes-1
        start = 0
        while self.Degree(start)==self.max_degree:
            start += 1                                                # Prima o poi trovo indice da cui partire con grado minore di max_degree (Sicuramente c'è perchè ho controllato che grafo non sia regolare)
        
        distance_array = self.span_tree.Dijkstra(start)                    # Meglio usare Dijkstra di Path_Len... NB: DISTANZE CALCOLATE SULLO SPANNING TREE!!!
        #print(distance_array, self.vertex_order)
        Merge_Sort(self.vertex_order, distance_array,0,self.n_nodes-1)     # In vertex_order ho l'ordine degli indici dal più lontano al più vicino a start!
        #print(distance_array, self.vertex_order)



    # ************ COLORAZIONE GREEDY **************

    def Greedy_Coloring(self, filename):
        self.Vertex_Ordering()         # Ordino i vertici

        self.vertex_color = np.full(self.n_nodes, 'null', dtype=object)        # Colore ad ogni vertice. Inizialmente vuota.
        colors = np.genfromtxt(filename, dtype='str')
        if len(colors) < self.max_degree:
            print("Non ci sono abbastanza colori. Il programma non può continuare")
            sys.exit()

        colors_used = 0
        for index in self.vertex_order:                     # Itero sul numero di vertici nell'ordine trovato in precedenza
            k = 0                          
            while self.Color_Already_Used(colors[k],index): # Se colore colors[k] già usato nei vertici adiacenti a quelli con indice vertex_order[i] vado avanti
                k += 1
            if k > colors_used:
                colors_used = k
            self.vertex_color[index] = colors[k]            # Attenzione: ho colorato il vertice numero vertex_order[i]!
        return colors_used+1

    def Color_Already_Used(self, color, index):
        for j in range(self.n_nodes):               # Riga fissata da index, scorro solo sulle colonne j
            if self.adj_matrix[index][j] == 1 and self.vertex_color[j] == color: # Se esiste un collegamento tra index e j e se j ha il colore già usato...
                return True
        return False




    # ********** DIJKSTRA ***********

    def Dijkstra(self, start):                              # Restituisce vettore con i cammini minimi da start a tutti gli altri
        distance_array = np.zeros(self.n_nodes, dtype=int)
        V = np.arange(self.n_nodes)                         # Insieme dei vertici che non appartengono ad S
        V = np.delete(V, start)
        S = np.array([start])                               # Inizialmente S contiene solo il vertice con indice "start"
        
        for i in range(self.n_nodes):
            if self.adj_matrix[i][start] == 1:
                distance_array[i] = 1                       # Se c'è un collegamento diretto: salvo distanza 1
            elif i!=start:
                distance_array[i] = self.n_nodes +1         # Collegamenti saranno lunghi al massimo n_nodes... dove non ci sono collegamenti metto n_nodes + 1
            else:
                distance_array[i] = 0                       # Se mi trovo in corrispondenza della sorgente metto 0

        while V.size != 0:                                  # Continuo fino a quando V non è vuoto
            index_min = V[0]                                # distanza_min = distance_array[index_min], inizalmente corrispondente al primo elemento di V...
            
            for index in V: 
                if distance_array[index] < distance_array[index_min]:
                    index_min = index                       # ... trovo indice in V che ha distanza minima da start
            
            S = np.append(S, index_min)
            V = np.delete(V, np.argwhere(V==index_min))     # Aggiungo nuovo elemento ad S e lo elimino da V
            
            for index in V:                                 # Per ogni vertice prendo il minimo tra la distanza calcolata in precedenza e il nuovo percorso che passa dall'elemento in V con distanza minima per poi collegarsi ad index con un solo passo
                if self.adj_matrix[index_min][index] == 1:  # NB: SOLO SE ESISTE UN COLLEGAMENTO TROVO IL MINIMO!!!!!
                    distance_array[index] = min(distance_array[index], distance_array[index_min] + 1)
        
        return distance_array
            




    # ************ METODI DI CHECK ************


    def Calculate_Edges(self):
        m_edges = 0
        for i in range(self.n_nodes):
            for j in range(i+1,self.n_nodes):     # Controllo solo metà matrice senza diagonale, se non è ben costruita sarà evidente dal Check
                if self.adj_matrix[i][j] == 1:
                    m_edges += 1
        return m_edges


    def Symmetry_Matrix(self):
        if self.n_nodes != len(self.adj_matrix[0]):                     # Se dimensioni diverse non posso correggere!
            return False
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):                          # Mi basta controllare metà matrice, esclusa la diagonale
                if self.adj_matrix[i][j] != self.adj_matrix[j][i]:      # Se esiste collegamento in una sola direzione aggiungo anche collegamento nell'altra
                    self.adj_matrix[i][j] = 1
                    self.adj_matrix[j][i] = 1
                    print( "Un collegamento mancante tra i vertici ", i, " e ", j, " è stato corretto")
        return True


    def Loop_Matrix(self):                                             # Posso sempre correggere, elimino loop!
        for i in range(self.n_nodes):
            if self.adj_matrix[i][i] == 1:
                self.adj_matrix[i][i] = 0
                print("E' stato eliminato il loop al vertice ", i)


    def Regular_Matrix(self):
        grado = self.Degree(0)  # Salvo il grado del primo vertice. Se tutti uguali restituisco True
        for i in range(1,self.n_nodes):
            if grado != self.Degree(i):
                return False
        return True


    def Connected_Matrix(self):
        distance_array = self.Dijkstra(0)                # Uso algoritmo di Dijkstra per trovare vettore con distanze da d
        for d in distance_array:
            if d > self.n_nodes:                         # Cammino di lunghezza massima: n_nodes (da implementazione Dijkstra, due nodi non collegati presentano distanza n+1)
                return False
        return True



    # ********* GRADI E CAMMINI ***********

    def Degree(self, i):
        grado = 0
        for j in range(self.n_nodes):
            if self.adj_matrix[i][j] == 1:
                grado += 1
        return grado


    def Max_Degree(self):   
        grado_max = self.Degree(0)                 # Salvo il grado del primo nodo
        for i in range(1, self.n_nodes):
            grado = self.Degree(i)
            if grado > grado_max:
                grado_max = grado
        return grado_max


    def Path_Len(self, i, j):                              # Esiste un cammino tra i e j? 
        if i==j:                                     
            return 0
        m = np.copy(self.adj_matrix)
        len = 0
    
        for len in range(1, self.n_nodes):                # len è lunghezza del path minimo!
            if m[i][j] >0:
                return len
            m = np.matmul(m,self.adj_matrix)
        return 0


    # ********* PRINT **********

    def Print(self):
        np.savetxt('Data/adj_matrix_finale.txt', self.adj_matrix, fmt="%i") 
        np.savetxt('Data/adj_matrix_tree.txt', self.span_tree.adj_matrix, fmt="%i") 
        np.savetxt('Data/vertex_coloring.txt', self.vertex_color, fmt="%s") 

        G = nx.from_numpy_matrix(self.adj_matrix)                      # Creo Grafo con libreria Networkx a partire dalla matrice di adiacenza
        G_t = nx.from_numpy_matrix(self.span_tree.adj_matrix)          # Albero di supporto creato con Depth_First_Search
        position = nx.spring_layout(G)

        new_labels = {} 
        original_labels = {}                                                   # Creo dizionario con nuovo ordine
        for i in range(self.n_nodes):
            new_labels[self.vertex_order[i]] = i                           # vertex_order racchiude i vertici da colorare in maniera ordinata... se vertex_order[0]=6 vuol dire che il vertice 6 è il primo vertice!
            original_labels[i] = i

        fig, axes = plt.subplots(2,2, figsize=(15,15))
        ax = axes.flatten()
        
        ax[0].set_title("Grafo Originale")
        nx.draw(G, position, node_color='b', with_labels=False, node_size=120, ax=ax[0])   # Stampo Grafo originale (non colorato, ordine originale)
        nx.draw_networkx_labels(G,position, original_labels, font_size=8, font_color="whitesmoke", ax=ax[0])

        ax[1].set_title("Albero di Supporto con Ordinamento Originale")
        nx.draw(G_t, position, node_color='g', with_labels=False, node_size=120, ax=ax[1])   # Stampo albero di supporto e con ordinamento originale
        nx.draw_networkx_labels(G_t,position, original_labels, font_size=8, font_color="whitesmoke", ax=ax[1])

        ax[2].set_title("Albero di Supporto con Nuovo Ordinamento")
        nx.draw(G_t, position, node_color='g', with_labels=False, node_size=120, ax=ax[2])   # Stampo albero di supporto e con ordinamento originale
        nx.draw_networkx_labels(G_t,position, new_labels, font_size=8, font_color="whitesmoke", ax=ax[2])

        ax[3].set_title("Grafo Colorato con Nuovo Ordinamento")
        nx.draw(G, position, node_color=self.vertex_color, with_labels=False, node_size=120, ax=ax[3])   # Stampo Grafo colorato e con ordinamento dato dal Vertex Ordering
        nx.draw_networkx_labels(G,position, new_labels, font_size=8, font_color="whitesmoke", ax=ax[3])
        
        fig.savefig("Data/colored_graph.png")

        plt.show(block=True)        







# ********* FUNZIONI DI ORDINAMENTO ************


def Merge_Sort(vertex_list, distance_list, p, r):
    if p<r:
        q = int((p+r)/2)    # Punto intermedio tra p ed r
        Merge_Sort(vertex_list, distance_list,p,q)          # ACHTUNG: limiti sono sempre compresi!
        Merge_Sort(vertex_list, distance_list,q+1,r)
        Merge(vertex_list, distance_list, p,q,r)

def Merge(vertex_list, distance_list, p,q,r):
    n1 = q-p+1               # Lunghezza vettore di sx 
    n2 = r-q                 # Lunghezza vettore di dx 
    L = np.zeros(n1+1, dtype=int)       # Vettori in cui salvo le distanze
    R = np.zeros(n2+1, dtype=int)
    L_v = np.zeros(n1+1, dtype=int)     # Vettori in cui salvo gli indici dei vertici
    R_v = np.zeros(n2+1, dtype=int)
    
    L[n1] = -1
    R[n2] = -1    # Ultimi due elementi posti a -1 (tanto distanze tutte positive...)
    
    for i in range(n1):
        L[i] = distance_list[p+i]       # A sinistra di q (compreso) ...
        L_v[i] = vertex_list[p+i]
    for j in range(n2):
        R[j] = distance_list[q+j+1]     # ... e a destra di q
        R_v[j] = vertex_list[q+j+1]

    i=0
    j=0
    for k in range(p,r+1):
        if L[i] >= R[j]:                # ACHTUNG: ORDINE DECRESCENTE!!!!!!!!!
            distance_list[k] = L[i]
            vertex_list[k] = L_v[i]
            i += 1
        else:
            distance_list[k] = R[j]
            vertex_list[k] = R_v[j]
            j += 1






'''
def Cycle_Matrix(adj_matrix, n_nodes):          # Sbagliato! Un percorso potrebbe anche tornare indietro!
    for i in range(n_nodes):
        if Path_Len(adj_matrix, n_nodes, i, i) > 0:     # Se trovo un ciclo mi fermo!
            return True
    return False
'''                

'''                                             # Manca modo efficiente per trovare cicli, qui non è necessario trovare spanning tree MINIMO!
def Kruskal(self):
        # Non devo ordinare l'insieme degli archi perché grafo non è pesato!
        m = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):           # Considero solo metà della matrice di adiacenza (diagonale esclusa)
                if self.adj_matrix[i][j] == 1:           # Trovo gli archi in ordine di apparizione nella matrice di adiacenza
                    m[i][j] = 1
                    m[j][i] = 1
                    if Cycle_Matrix(m,self.n_nodes)==True:
                        m[i][j] = 0
                        m[j][i] = 0                     # Se con modifica creo un ciclo torno come prima...
        
        self.span_tree = Graph("",m)                   # Lo spanning Tree è un ulteriore grafo costruito passando la matrice di adiacenza al costruttore!!!
'''

'''                                                      # DIJKSTRA PIU' EFFICIENTE
    def Connected_Matrix(self):      # Posso usare funzione Path_Len preesistente ma poco efficiente...
        conn_matrix = np.identity(self.n_nodes, dtype=int)   # Matrice identità di dimensione nxn. Ogni elemento diventa 1 se e solo se esiste un collegamento tra quei due vertici
        m = np.copy(self.adj_matrix)

        for len in range(1, self.n_nodes):           # len è lunghezza del path minimo!
            for i in range(self.n_nodes):
                for j in range(i+1,self.n_nodes):            # Se controllo solo metà matrice non sarebbe necessario modificare anche il trasposto di conn_matrix (più sicuro)
                    if m[i][j] > 0:                     # Attenzione: m non ha entrate 0 o 1 perché sto usando moltiplicazione usuale!
                        conn_matrix[i][j] = 1           # Se trovo un collegamento lo salvo (non posso tornare a 0...)
                        conn_matrix[j][i] = 1
            m = np.matmul(m,self.adj_matrix)

        for i in range(self.n_nodes):
            for j in range(i+1,self.n_nodes):         # Controllo solo metà matrice
                if conn_matrix[i][j] == 0:
                    return False                 # Se è presente un'entrata pari a 0 il grafo non è connesso
        return True
    '''
