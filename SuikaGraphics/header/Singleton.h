#pragma once
#include <iostream>
template <typename T>
class Singleton
{
public:
    ~Singleton() {
        std::cout << "destructor called!" << std::endl;
    }
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    static T& get_instance() {
        static T instance;
        return instance;
    }
private:
    // Set constructor as private
    Singleton() {
        std::cout << "constructor called!" << std::endl;
    }
};