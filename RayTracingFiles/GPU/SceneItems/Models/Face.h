//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_FACE_H
#define GRAPHICS_FACE_H

#include <vector>
#include <iostream>

#include "../../Matrix/Vector.h"

class Face {

public:

    Face() = default;
    Face(const Face &) = default;
    Face &operator=(const Face &) = default;
    virtual ~Face()= default;

    __host__ explicit Face(const std::vector<int> &, int);

    std::vector<int> vertexIndexList;
    int materialIndex = 0;
    std::vector<Vector> normals;
    
    Vector columnOne, columnTwo;
    bool calcColumns = false;
};

__host__ std::ostream &operator<<(std::ostream &, const Face &);


#endif //GRAPHICS_FACE_H
