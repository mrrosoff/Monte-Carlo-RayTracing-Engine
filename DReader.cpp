//
// Created by Max Rosoff on 9/7/2019.
//

#include "DReader.h"

using namespace std;

DReader &DReader::operator<<(const string &driverFile)
{
    readDriver(driverFile);
}

void DReader::readDriver(const string &driverFile)
{
    ifstream driverReader(driverFile);

    if (!driverReader)
    {
        string err = strerror(errno);
        throw invalid_argument("Failure to open Driver File - " + driverFile + ": " + err);
    }

    string driverLine;

    while(getline(driverReader, driverLine))
    {
        stringstream lineReader(driverLine);
        string token;

        vector<string> lineData;

        while(getline(lineReader, token, ' '))
        {
            lineData.push_back(token);
        }

        if(lineData[0] == "eye")
        {
            eye << stod(lineData[1]),
                   stod(lineData[2]),
                   stod(lineData[3]);
        }

        if(lineData[0] == "look")
        {
            lookAtPoint << stod(lineData[1]),
                           stod(lineData[2]),
                           stod(lineData[3]);
        }

        if(lineData[0] == "up")
        {
            upVector << stod(lineData[1]),
                        stod(lineData[2]),
                        stod(lineData[3]);
        }

        if(lineData[0] == "d")
        {
            focalLength =  stod(lineData[1]);
        }

        if(lineData[0] == "bounds")
        {
            bounds << stod(lineData[1]),
                      stod(lineData[2]),
                      stod(lineData[3]),
                      stod(lineData[4]);
        }

        if(lineData[0] == "res")
        {
            resolution << stod(lineData[1]),
                          stod(lineData[2]);
        }

        if(lineData[0] == "ambient")
        {
            ambientLight << stod(lineData[1]),
                            stod(lineData[2]),
                            stod(lineData[3]);
        }

        if(lineData[0] == "light")
        {
            Eigen::Vector3d position;
            Eigen::Vector3d rgb;
            double w;

            position << stod(lineData[1]),
                        stod(lineData[2]),
                        stod(lineData[3]);

            w = stod(lineData[4]);

            rgb << stod(lineData[5]),
                   stod(lineData[6]),
                   stod(lineData[7]);

            lights.emplace_back(position, rgb, w);
        }

        if(lineData[0] == "sphere")
        {
            Eigen::Vector3d position;
            double radius;
            Eigen::Vector3d Ka;
            Eigen::Vector3d Kd;
            Eigen::Vector3d Ks;
            Eigen::Vector3d Kr;

            position << stod(lineData[1]),
                        stod(lineData[2]),
                        stod(lineData[3]);

            radius = stod(lineData[4]);

            Ka << stod(lineData[5]),
                  stod(lineData[6]),
                  stod(lineData[7]);

            Kd << stod(lineData[8]),
                  stod(lineData[9]),
                  stod(lineData[10]);

            Ks << stod(lineData[11]),
                  stod(lineData[12]),
                  stod(lineData[13]);

            Kr << stod(lineData[14]),
                  stod(lineData[15]),
                  stod(lineData[16]);

            vector<Eigen::Vector3d> colors = {Ka, Kd, Ks, Kr};

            spheres.emplace_back(position, colors, radius);
        }
    }
}
