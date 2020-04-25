#include <iostream>

template<typename... Args>
bool checkSizes(Args const&... args)
{
    return (... == args.size());
}

template<typename First, typename... Args>
void printCSVLine(std::ofstream& file, std::string const& seperator, First const& first, Args const&... args)
{
    auto print_with_seperator = [](std::ofstream& file, const std::string& seperator, const auto& arg) -> const auto& {
        file << seperator;
        return arg;
    };
    
    std::cout << first;
    (file << ... << print_with_seperator(file, seperator, args)) << std::endl;
}