//
// Created by Max Rosoff on 9/7/2019.
//

#include "Writer.h"

using namespace std;

Writer::Writer(const std::string &driverName) : driverName(driverName) {}

int Writer::writeObject(const string &objName, const vector<string> &lines)
{
    string outFileName;

    if(writtenFiles[objName] < 10)
    {
        outFileName = objName + "_mw" + to_string(0) + to_string(writtenFiles[objName]) + ".obj";
    }

    else
    {
        outFileName = objName + "_mw" + to_string(writtenFiles[objName]) + ".obj";
    }

    ofstream objOutputFile("./" + driverName + "/" + outFileName);

    if (!objOutputFile)
    {
        string err = strerror(errno);
        cerr << "Failure to open Obj File - " << objName <<  ": " << err <<'\n';
        return 1;
    }

    else
    {
        for(const auto &line : lines)
        {
            objOutputFile << line + '\n';
        }
    }

    return 0;
}

int Writer::writeTxt(const string &objName, const Eigen::Matrix4d &tMatrix, const double absTrans, const double absTransInv) {

    string outFileName;

    if(writtenFiles[objName] < 10)
    {
        outFileName = objName + "_transform_mw" + to_string(0) + to_string(writtenFiles[objName]) + ".txt";
    }

    else
    {
        outFileName = objName + "_transform_mw" + to_string(writtenFiles[objName]) + ".txt";
    }

    writtenFiles[objName] += 1;

    ofstream objOutputFile("./" + driverName + "/" + outFileName);

    if (!objOutputFile)
    {
        string err = strerror(errno);
        cerr << "Failure to open Obj File for reading! - " << objName <<  ": " << err <<'\n';
        return 1;
    }

    else
    {
        Eigen::IOFormat noAlign(Eigen::StreamPrecision, Eigen::DontAlignCols);

        objOutputFile << fixed << setprecision(3)
                      << "# Transformation matrix" << '\n'
                      << tMatrix.format(noAlign) << '\n'
                      << '\n'
                      << "# Inverse transformation matrix" << '\n'
                      << tMatrix.inverse().format(noAlign) << '\n'
                      << '\n'
                      << setprecision(10)
                      << "# Sum absolute translations from original to transformed" << '\n'
                      << absTrans << '\n'
                      << '\n'
                      << "# Sum absolute translations from original to transformed to \"original\"" << '\n'
                      << absTransInv << '\n'
                      << '\n';
    }

    return 0;
}
