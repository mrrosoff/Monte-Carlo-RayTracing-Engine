//
// Created by Max Rosoff on 10/19/2019.
//

#include "Material.h"

using namespace std;

Material::Material(const string &name, const Eigen::Vector3d &Ka, const Eigen::Vector3d &Kd, const Eigen::Vector3d &Ks, const Eigen::Vector3d &Kr, const double Ns, const double Ni, const double d, const int illum) :

name(name), Ka(Ka), Kd(Kd), Ks(Ks), Kr(Kr), Ns(Ns), Ni(Ni), d(d), illum(illum)

{}

ostream &operator<<(ostream &out, const Material &mat)
{
    Eigen::IOFormat ArrayFormat(Eigen::StreamPrecision, 0, "", ", ", "", "", "[", "]");

    out << "Material Name: " << mat.name << '\n';
    out << "Material Ka: " << mat.Ka.format(ArrayFormat) << '\n';
    out << "Material Kd: " << mat.Kd.format(ArrayFormat) << '\n';
    out << "Material Ks: " << mat.Ks.format(ArrayFormat) << '\n';
    out << "Material Kr: " << mat.Kr.format(ArrayFormat) << '\n';
    out << "Material Ns: " << mat.Ns << '\n';
    out << "Material Ni: " << mat.Ni << '\n';
    out << "Material D: " << mat.d << '\n';
    out << "Material Illum: " << mat.illum << '\n';

    return out;
}
