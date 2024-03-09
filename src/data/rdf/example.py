#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdflib import Graph,URIRef, Literal

def v(s):
    return '<' + str(s) + '>'

g = Graph()

p = URIRef("http://dbpedia.org/ontology/capital")
qres = g.query(
    """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX db: <http://dbpedia.org/resource/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>

    SELECT *
    WHERE {
      SERVICE <https://dbpedia.org/sparql> {
        ?country rdf:type <http://schema.org/Country> ;
                 dbo:capital ?capital .
      }
    }
    """
)

for row in qres:
   g.add([row.country, p, row.capital])
#print(g.serialize())

[[str(s), str(p), str(o)] for [s, p, o] in g]