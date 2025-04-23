#include <gtest/gtest.h>

#include "arrays.hpp"

TEST(ArraysTest, IsCloseTrue) {
    double a = 0.123456;
    double b = 0.123457;
    EXPECT_TRUE(isClose(a, b, 1e-5));
}

TEST(ArraysTest, IsCloseFalse) {
    double a = 0.123;
    double b = 0.124;
    EXPECT_FALSE(isClose(a, b, 1e-5));
}