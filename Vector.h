//
// Created by clash on 11/06/2023.
//

#ifndef AILEARNING_VECTOR_H
#define AILEARNING_VECTOR_H

#include <initializer_list>
#include <iostream>
#include <vector>

namespace Wuu {
    template<typename T>
    class Vector {
    public:

        using container = std::vector<T>;
        using iterator = typename container::iterator;
        using const_iterator = typename container::const_iterator;

        iterator begin() { return m_values.begin(); }
        iterator end() { return m_values.end(); }
        const_iterator begin() const { return m_values.begin(); }
        const_iterator end() const { return m_values.end(); }

        Vector() = default;

        explicit Vector(std::size_t size)
        {
            m_values.resize(size);
            for(auto& element : m_values)
            {
                element = 0;
            }
        }
        Vector(std::initializer_list<T> list)
        {
            m_values = list;
        }

        Vector(const Vector<T>& other)
        {
            m_values.resize(other.m_values.size());
            m_values = other.m_values;
        }

        Vector(Vector<T>& other)
        {
            m_values.resize(other.m_values.size());
            m_values = other.m_values;
        }

        Vector(std::vector<T> values)
        {
            m_values = values;
        }

        [[nodiscard]] std::size_t getSize() const noexcept
        {
            return m_values.size();
        }

        void setSize(std::size_t size)
        {
            m_values.resize(size);
            for(auto& element : m_values)
            {
                element = 0;
            }
        }

        auto& getValues() noexcept
        {
            return m_values;
        }

        friend auto& operator<<(std::ostream& os, const Vector<T>& vector)
        {
            for(const auto& value : vector)
            {
                os << " " << value << " ";
            }
            return os;
        }

        T& operator[](int index)
        {
            return m_values[index];
        }

        const T& operator[](int index) const
        {
            return m_values[index];
        }

        Vector<T>& operator=(const Vector<T>& other)
        {
            if(this == &other)
                return *this;

            m_values.resize(other.m_values.size());
            m_values = other.m_values;
        }

    private:

        container m_values;

    };
}

#endif //AILEARNING_VECTOR_H
