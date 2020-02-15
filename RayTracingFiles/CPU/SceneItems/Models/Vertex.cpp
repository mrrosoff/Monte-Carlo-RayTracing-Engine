//
// Created by Max Rosoff on 9/7/2019.
//

#include "Vertex.h"

using namespace std;

Vertex::Vertex(const Vector &vertex) :

vertex(vertex)

{}

ostream &operator<<(ostream &out, const Vertex &vertex)
{
    out << vertex.vertex;

    return out;
}
