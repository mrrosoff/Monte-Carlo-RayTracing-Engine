//
// Created by Max Rosoff on 9/21/2019.
//

#include "PWriter.h"

using namespace std;

PWriter::PWriter(const string &outFile) :

outFile(outFile)

{}

PWriter& PWriter::operator<<(const vector<vector<vector<int>>> &img)
{
    ofstream outWriter(outFile);

    if (!outWriter)
    {
        string err = strerror(errno);
        throw invalid_argument("Failure to open Output File - " + outFile + ": " + err);
    }

    auto height = img.size();
    auto width = img[0].size();

    outWriter << "P3" << '\n';
    outWriter << height << " " << width << " " << 255 << '\n';

    for(size_t i = 0; i < height; i++)
    {
        for(size_t j = 0; j < width; j++)
        {
            outWriter << img[i][j][0] << " ";
            outWriter << img[i][j][1] << " ";

            if(j != width - 1)
            {
                outWriter << img[i][j][2] << " ";
            }

            else
            {
                outWriter << img[i][j][2] << "\n";
            }
        }
    }

     return *this;
}
