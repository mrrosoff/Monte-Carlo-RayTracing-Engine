//
// Created by Max Rosoff on 9/7/2019.
//

#include "Vertex.h"

using namespace std;

Vertex::Vertex(const Eigen::Vector3d &vertex) :

vertex(vertex)

{}

ostream &operator<<(ostream &out, const Vertex &vertex)
{
    return out;
}
