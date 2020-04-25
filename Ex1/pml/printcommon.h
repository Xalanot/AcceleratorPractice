#include <iostream>

template<typename First>
auto checkSizes(First const& first)
{
    return first.size();
}

template<typename First, typename... Args>
auto  checkSizes(First const& first, Args const&... args)
{
    return first.size() == checkSizes(args...);
}

template<typename First>
auto printCSVLine(std::ofstream& file, std::string const& seperator, First const& first)
{
    return first;
}

template<typename First, typename... Args>
auto  printCSVLine(std::ofstream& file, std::string const& seperator, First const& first, Args const&... args)
{
    file << first << seperator << printCSVLine(file, seperator, args...);
    file << std::endl;
}