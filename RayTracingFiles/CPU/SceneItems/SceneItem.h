//
// Created by Max Rosoff on 11/17/2019.
//

#ifndef RAYTRACER_SCENEITEM_H
#define RAYTRACER_SCENEITEM_H

#include <exception>

#include "../Matrix/Vector.h"
#include "../RayTracing/Ray.h"
#include "Material.h"

class SceneItem {

public:

    virtual bool intersectionTest(Ray &) const = 0;
    virtual Ray makeExitRefrationRay(const Ray &, double, double) const = 0;
    Vector doSnellsLaw(const Vector &, const Vector &, double, double) const;
};

#endif //RAYTRACER_SCENEITEM_H
