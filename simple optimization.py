import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import networkx as nx


def find_optimal_y(G, d):
    """
    funkcja find_optimal_y przyjmuje dwa parametry
    G - pewien graf G - obiekt klasy nx.Graph()
    d - wartość określającą na ile wymiarów ma być rzutowany wynik

    funkcja w wyniku zwraca macierz Y rozmiaru dxn, gdzie n to liczba wierzchołków grafu,
    która powinna jak najlepiej odwzorowywać graf G w d wymiarach

    na przykład:
    Cykl długości 3 jest dobrze reprezentowany przez macierz
    Y = {{-1, 0, 1}, {-1, 1, 0}}
    """

    def cond(Y):
        """
        funkcja sprawdza warunek, który powinnna spełniać szukana macierz Y. Y^T @ D @ Y = I
        czyli innymi słowy iloczyn macierzy Y transponowane, macierzy stopni D, oraz Y, powinien być macierzą jednostkową

        funkcja zwraca 0 jezeli warunek jest spelniony, odchylenie przecietne roznicy macierzy w przeciwnym wypadku
        """
        Y = np.reshape(Y, (n, d)) # dzięki tej operacji scipy.optimize moze pracowac na macierzy o kształcie jednowymiarowym
        A = np.matmul(np.matmul(np.transpose(Y), D), Y)
        B = np.identity(d) # macierz jednostkowa o rozmiarze takim jak A
        if np.allclose(A, B):
            return 0
        return np.mean(np.abs(A-B))

    def y_trace(Y):
        """
        funkcja do zminimalizowania
        nalezy znalezc taką macierz Y spełniającą warunek cond, aby ślad iloczynu macierzy Y^T, L, Y byl jak najmniejszy
        (tzn minimalizujemy tr(Y^T @ L @ Y)
        """
        Y = np.reshape(Y, (n, d)) # dzięki tej operacji scipy.optimize moze pracowac na macierzy o kształcie jednowymiarowym
        return np.trace(np.matmul(np.matmul(np.transpose(Y), L), Y)) # zwracamy tr(Y^T @ L @ Y)

    # przygotowanie danych

    # znajdujemy odpowiednie macierze
    # macierz sąsiedztwa G
    A = nx.to_scipy_sparse_array(G).toarray()
    # laplasjan G
    L = nx.laplacian_matrix(G).toarray()
    # macierz stopni G
    D = L + A
    #liczba wierzchołków grafu G
    n = G.number_of_nodes()

    cons = [{'type': 'eq', 'fun': cond}] # constraints na funkcje
    Y = np.random.rand(n * d) # wybranie macierzy Y jako losowej - to moze pomoze opt.minimize

    res = opt.minimize(y_trace, Y, method="SLSQP", constraints=cons, options={"maxiter": 50000})

    print(res)
    Y = np.reshape(res.x, (d, -1))

    return Y


np.random.seed(44)
# przy uzyciu networkx stworzymy sobie grafy do testow
cycle_graph = nx.cycle_graph(20) # cykl długosci 5
barbell_graph = nx.barbell_graph(30, 5) # graf sztangowy
complete_graph = nx.complete_graph(5) # graf pelny


# wizualizujemy graf za pomocą matplotlib
nx.draw_networkx(barbell_graph)
plt.show()

Y = find_optimal_y(barbell_graph, 2)

plt.scatter(Y[0], Y[1])
plt.show()

