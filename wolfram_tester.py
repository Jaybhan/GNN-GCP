"""from wolframclient.language import wl          # Wolfram symbolic constructors
from wolframclient.evaluation import WolframLanguageSession

with WolframLanguageSession() as session:
    # 1. Build the graph entirely on the Python side
    g = wl.Graph([
        wl.UndirectedEdge(1, 2),
        wl.UndirectedEdge(2, 3),
        wl.UndirectedEdge(3, 1),
        wl.UndirectedEdge(3, 4)
    ])

    # 2. Send the HadwigerNumber expression to Wolfram **for evaluation**
    hadwiger = session.evaluate(wl.HadwigerNumber(g))

    # 3. Print the numeric result
    print("Hadwiger number of the graph:", hadwiger)
"""

from wolframclient.language import wl
from wolframclient.evaluation import WolframLanguageSession

with WolframLanguageSession() as s:
    s.evaluate(wl.Needs("Combinatorica`"))          # ① load package

    g = s.evaluate(wl.FromUnorderedPairs([[1,2],     # ② build a Combinatorica graph
                                          [2,3],
                                          [3,1],
                                          [3,4]]))

    h = s.evaluate(wl.HadwigerNumber(g))            # ③ call the function
    print(h)                                        # -> 3
