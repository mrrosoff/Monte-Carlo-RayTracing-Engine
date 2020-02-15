//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_VERTEX_H
#define GRAPHICS_VERTEX_H

#include <vector>
#include <iostream>

#include "../../Matrix/Vector.h"

class Vertex {

public:

    Vertex() = default;
    Vertex(const Vertex &) = default;
    Vertex &operator=(const Vertex &) = default;
    virtual ~Vertex()= default;

    explicit Vertex(const Vector &);

    Vector vertex;
    std::vector<int> adjacentFaces;
};

std::ostream &operator<<(std::ostream &, const Vertex &);


#endif //GRAPHICS_VERTEX_H
