//
// Created by Max Rosoff on 10/19/2019.
//

#include "Material.h"

using namespace std;

Material::Material(const string &name, const Eigen::Vector3d &albedo, const int materialType) :

name(name), albedo(albedo)

{
    if(materialType == 1)
    {
        isLight = true;
    }

    else if(materialType == 2)
    {
        isMirror = true;
    }

    else if(materialType == 3)
    {
        isGlass = true;
    }
}

ostream &operator<<(ostream &out, const Material &mat)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");
    out << boolalpha;
    out << "Material Name: " << mat.name << '\n';
    out << "Material Albedo: " << mat.albedo.format(ArrayFormat) << '\n';
    out << "Material is Light: " << mat.isLight << '\n';
    out << "Material is Mirror: " << mat.isMirror << '\n';
    out << "Material is Glass: " << mat.isGlass << '\n';

    return out;
}
