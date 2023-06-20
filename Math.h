//
// Created by clash on 11/06/2023.
//

#ifndef AILEARNING_MATH_H
#define AILEARNING_MATH_H

#include "Vector.h"

using namespace Wuu;

class UnequalVectorLength : public std::exception {
public:
    explicit UnequalVectorLength(char* operation) : m_message(operation) {}

    char* what()
    {
        return m_message;
    }
private:

    char* m_message;
};

template<typename l, typename r>
void add(Vector<l>& left, const Vector<r>& right)
{
    if(left.getSize() != right.getSize()) throw UnequalVectorLength("Addition");
    for(int i = 0; i < left.getSize(); i++)
    {
        left[i] += right[i];
    }
}

template<typename l, typename r>
[[nodiscard]] Vector<l> addR(const Vector<l>& left,const Vector<r>& right)
{
    if(left.getSize() != right.getSize()) throw UnequalVectorLength("Addition");
    Vector<l> sum;
    sum.setSize(left.getSize());
    for(int i = 0; i < left.getSize(); i++)
    {
        sum[i] = left[i] + right[i];
    }
    return sum;
}

template<typename l, typename r>
void subtract(Vector<l>& left, const Vector<r>& right)
{
    if(left.getSize() != right.getSize()) throw UnequalVectorLength("Subtraction");
    for(int i = 0; i < left.getSize(); i++)
    {
        left[i] -= right[i];
    }
}

template<typename l, typename r>
[[nodiscard]] Vector<l> subtractR(const Vector<l>& left, const Vector<r>& right)
{
    if(left.getSize() != right.getSize()) throw UnequalVectorLength("Subtraction");
    Vector<l> sum { left.getSize() };
    for(int i = 0; i < left.getSize(); i++)
    {
        sum[i] = left[i] - right[i];
    }
    return sum;
}

template<typename l, typename r>
void scale(Vector<l>& left, const r& right)
{
    for(int i = 0; i < left.getSize(); i++)
    {
        left[i] *= right;
    }
}

template<typename l, typename r>
[[nodiscard]] auto scaleR(const Vector<l>& left, const r& right)
{
    Vector<l> sum { left.getSize() };
    for(int i = 0; i < left.getSize(); i++)
    {
        sum[i] = left[i] * right;
    }
    return sum;
}

template<typename l, typename r>
[[nodiscard]] auto dot(const Vector<l>& left, const Vector<r>& right)
{
    if(left.getSize() != right.getSize()) throw UnequalVectorLength("Dot Product");
    float sum = 0.0f;
    for(int i = 0; i < left.getSize(); i++)
    {
        sum += left[i] * right[i];
    }
    return sum;
}

#endif //AILEARNING_MATH_H
