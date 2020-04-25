#include <iostream>
#include "csvwriter.h"
#include "measurement.h"

int main()
{
    CSVWriter csvwriter("test.csv");
    
    std::vector<std::string> headerNames{"sizes", "time"};
    csvwriter.setHeaderNames(std::move(headerNames));
    
    std::vector<int> sizes = {10, 20};
    std::vector<float> times = {2.4, 8.9};

    csvwriter.write(sizes, times);

    return 0;
}