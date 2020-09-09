//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_VERTEX_H
#define GRAPHICS_VERTEX_H

#include <vector>

#include "../../Matrix/Vector.h"

class Vertex {

public:

    Vertex() = default;
    Vertex(const Vertex &) = default;
    Vertex &operator=(const Vertex &) = default;
    virtual ~Vertex()= default;

    __host__ explicit Vertex(const Vector<3> &);

    Vector<3> vertex;
    std::vector<int> adjacentFaces;
};


#endif //GRAPHICS_VERTEX_H
