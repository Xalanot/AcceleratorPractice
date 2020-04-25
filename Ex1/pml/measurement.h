#include <chrono>

template<typename T>
class Measurement
{
public:
    void start()
    {
        startTime = std::chrono::high_resolution_clock::now();
    };
    
    void stop()
    {
        endTime = std::chrono::high_resolution_clock::now();       
    };
    
    double getTime() const
    {
        auto duration = std::chrono::duration_cast<T>(endTime - startTime);
        return duration.count();
    };
    
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Measurement<U>& measurement);
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Measurement<T>& measurement)
{
    os << measurement.getTime();
    return os;
}

template<typename T>
class MeasurementSeries
{
public:
    void start()
    {
        currentMeasurement.start();
    }

    void stop()
    {
        currentMeasurement.stop();
        measurements.push_back(currentMeasurement)
    }

    double getMeanTime()
    {
        double totalTime = 0;
        for (auto const& measurement: measurements)
        {
            totalTime += measurement.getTime();
        }

        return totalTime / measurements.size();
    }

    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const MeasurementSeries<U>& measurementSeries);

private:
    std::vector<Measurement<T>> measurements;
    Measurement<T> currentMeasurement;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const MeasurementSeries<T>& measurementSeries)
{
    os << measurementSeries.getMeanTime();
    return os;
}