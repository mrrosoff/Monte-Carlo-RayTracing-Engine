//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_DRIVERREADER_H
#define GRAPHICS_DRIVERREADER_H

#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>

#include <cerrno>
#include <cstring>

#include "Remap.h"


class DReader {

public:

    explicit DReader(const std::string &);

    DReader(const DReader &) = delete;
    DReader &operator=(const DReader &) = delete;
    ~DReader()= default;

    std::string driverName;
    std::vector<Remap> remaps;

private:

    std::string findDriverName(const std::string &);
    int readDriver();

    std::string driverPath;

};


#endif //GRAPHICS_DRIVERREADER_H
