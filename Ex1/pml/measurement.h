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
    
    double getTimeInMilliseconds() const
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
    os << measurement.getTimeInMilliseconds();
    return os;
}