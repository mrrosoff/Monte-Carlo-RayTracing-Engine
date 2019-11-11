
//
// Created by Max Rosoff on 9/2/2019.
//

#include <sys/stat.h>

#include "DReader.h"
#include "OReader.h"
#include "Writer.h"

using namespace std;

// This function finds the type of line specified in the .obj file.

// 0 -> Vertex
// 1 -> Vertex Normal
// 2 -> Face
// 3 -> Other

int findLineType(const string &line)
{
    if(line[0] == 'v')
    {
        if(line[1] == 'n')
        {
            return 1;
        }

        else
        {
            return 0;
        }
    }

    else if(line[0] == 'f')
    {
        return 2;
    }

    else
    {
        return 3;
    }
}

vector<double> breakLine(const string &line)
{
    stringstream tokenizer(line);
    string token;

    vector<double> vertex;

    while(getline(tokenizer, token, ' '))
    {
        if(token == "v")
        {
            continue;
        }

        vertex.push_back(stod(token));
    }

    return vertex;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        cout << "Usage: ./modeltoworld" << " " << "[Driver File]" << '\n';
        return 1;
    }

    // Initialize Driver Reader Object. This will create a list of transform objects.
    // Each Transform object holds a transformation matrix. This needs to be multiplied into the .obj file specified.

    DReader driver(argv[1]);

    // Create a new directory for all new files.

    int status = mkdir(driver.driverName.c_str(), S_IRWXU);

    // Initialize Object Writer Object. This will create and write to the correctly named file.

    Writer objOut(driver.driverName);

    for(const auto &a : driver.remaps) {

        OReader object(a.objPath);

        vector<string> newLines;

        double absTrans = 0;
        double absTransInv = 0;

        for (const auto &line : object.lines)
        {
            int type = findLineType(line);

            if(type == 0)
            {
                vector<double> vertex = breakLine(line);

                Eigen::Vector4d beforeVector;

                beforeVector << vertex[0], vertex[1], vertex[2], 1;

                auto afterVector = a.transformation * beforeVector;

                absTrans += abs(beforeVector[0] - afterVector[0]) +
                            abs(beforeVector[1] - afterVector[1]) +
                            abs(beforeVector[2] - afterVector[2]);

                auto reverseVector = a.transformation.inverse() * afterVector;

                absTransInv += abs(afterVector[0] - reverseVector[0]) +
                            abs(afterVector[1] - reverseVector[1]) +
                            abs(afterVector[2] - reverseVector[2]);

                newLines.push_back("v "s +
                                  to_string(afterVector[0]) + " "s +
                                  to_string(afterVector[1]) + " "s +
                                  to_string(afterVector[2]));
            }

            else if(type == 2 || type == 3)
            {
                newLines.push_back(line);
            }
        }

        absTransInv -= absTrans;

        string objName = a.objPath.substr(0, a.objPath.find_last_of('.'));

        // Pass the Object Write the name of the .obj file and what to write.

        objOut.writeObject(objName, newLines);
        objOut.writeTxt(objName, a.transformation, absTrans, absTransInv);
    }
}

