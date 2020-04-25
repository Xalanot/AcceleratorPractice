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
    std::vector<Measurement<std::chrono::seconds>> times2;

    for (auto const& ele : sizes)
    {
        Measurement<std::chrono::seconds> measurement;
        measurement.start();
        std::this_thread::sleep_for(std::chrono::seconds(ele));
        measurement.stop();
        times.push_back((measurement));

        Measurement<std::chrono::seconds> measurement2;
        measurement2.start();
        std::this_thread::sleep_for(std::chrono::seconds(2*ele));
        measurement2.stop();
        times2.push_back((measurement2));
    }

    csvwriter.write(sizes, times, times2);

    return 0;
}