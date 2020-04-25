#include <chrono>
#include <iostream>
#include <thread>

#include "csvwriter.h"
#include "measurement.h"

int main()
{
    CSVWriter csvwriter("measurment.csv");
    
    std::vector<std::string> headerNames{"sizes", "time"};
    csvwriter.setHeaderNames(std::move(headerNames));
    
    std::vector<int> sizes {1, 2};
    std::vector<Measurement<std::chrono::seconds>> times;

    for (auto const& ele : sizes)
    {
        Measurement<std::chrono::seconds> measurement;
        measurement.start();
        std::this_thread::sleep_for(std::chrono::seconds(ele));
        measurement.stop();
        times.push_back(std::move(measurement));
    }

    csvwriter.write(sizes, times);

    return 0;
}