#include <fstream>
#include <string>
#include <vector>

#include "printcommon.h"

class CSVWriter
{
public:
    CSVWriter(std::string filename)
    : filename(filename),
    seperator(std::string(", "))
    {}
    
    void setHeaderNames(std::vector<std::string>&& headerVector)
    {
        headerNames = std::move(headerVector);
    }
    
    template <typename First, typename... Args>
    void write(First const& first, Args const&... args)
    {
        openFile();
        printHeader();
        
        for (size_t i = 0; i < first.size(); ++i)
        {
            printCSVLine(file, seperator, first[i], args[i]...);
            file << std::endl;
        }

        closeFile();
    }
private:
    void openFile()
    {
        file.open(filename, std::ios::out | std::ios::trunc);
    }

    void closeFile()
    {
        file.close();
    }

    void printHeader()
    {
        file << headerNames[0];
        for (size_t i = 1; i < headerNames.size(); ++i)
        {
            file << seperator << headerNames[i];
        }
        file << std::endl;
    }

    std::ofstream file;
    std::string filename;
    std::string seperator;
    std::vector<std::string> headerNames;
};