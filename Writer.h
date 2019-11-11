//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_OWRITER_H
#define GRAPHICS_OWRITER_H

#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <cerrno>
#include <cstring>
#include <unordered_map>

#include "../Eigen/Eigen/Eigen"

class Writer {

public:

    explicit Writer(const std::string &);

    Writer(const Writer &) = delete;
    Writer &operator=(const Writer &) = delete;
    ~Writer()= default;

    int writeObject(const std::string &, const std::vector<std::string> &);
    int writeTxt(const std::string &, const Eigen::Matrix4d &, double, double);

private:

    std::string driverName;
    std::unordered_map<std::string, int> writtenFiles;
};


#endif //GRAPHICS_OWRITER_H
