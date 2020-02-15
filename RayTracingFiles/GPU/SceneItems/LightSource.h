//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_LIGHTSOURCE_H
#define GRAPHICS_LIGHTSOURCE_H

#include <iostream>

#include "../Matrix/Vector.h"

class LightSource {

public:

    LightSource() = delete;
    LightSource(const LightSource &) = default;
    LightSource &operator=(const LightSource &) = delete;
    ~LightSource() = default;

    __host__ explicit LightSource(const Vector &, const Vector &, double);

    Vector position;
    Vector rgb;
    double w = 0;

};

__host__ std::ostream &operator<<(std::ostream &out, const LightSource &light);


#endif //GRAPHICS_LIGHTSOURCE_H
