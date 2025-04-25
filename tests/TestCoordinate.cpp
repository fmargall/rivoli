#include <gtest/gtest.h>

#include "coordinate.hpp"
#include "arrays.hpp"

TEST(CoordinateTest, Clone2D) {
    Coordinate2D<double> c(0.5, 1.0);
    std::unique_ptr<Coordinate> copy = c.clone();
    auto* down = dynamic_cast<Coordinate2D<double>*>(copy.get());

    ASSERT_NE(down, nullptr);
    EXPECT_TRUE(isClose(down->getTheta(), 0.5));
    EXPECT_TRUE(isClose(down->getPhi(), 1.0));
}

TEST(CoordinateTest, BilateralSymmetry2D) {
    Coordinate2D<double> c(0.3, 1.5);
    auto sym = c.getBilateralSymmetrical();
    EXPECT_TRUE(isClose(sym.getPhi(), 2 * glm::pi<double>() - 1.5));
}

TEST(CoordinateTest, ConvertRusinkiewiczToSpherical) {
    Coordinate3DRusinkiewicz<double> rus(0.2, 0.3, 1.2);
    Coordinate3DSpherical<double> sph = static_cast<Coordinate3DSpherical<double>>(rus);

    auto back = Coordinate3DRusinkiewicz<double>(
        rusinkiewiczCoordToStandardCoord(rus.getThetaH(), rus.getThetaD(), rus.getPhiD())
    );

    // Round-trip inverse check
    EXPECT_TRUE(isClose(sph.getThetaI(), static_cast<Coordinate3DSpherical<double>>(rus).getThetaI(), 1e-6));
}

TEST(CoordinateTest, ReciprocalAndSymmetry) {
    Coordinate3DRusinkiewicz<double> coord(0.3, 0.4, 1.0);

    auto reciprocal = coord.getReciprocal();
    auto symmetrical = coord.getBilateralSymmetrical();

    EXPECT_TRUE(isClose(reciprocal.getPhiD() , coord.getPhiD() + glm::pi<double>()     , 1e-6));
    EXPECT_TRUE(isClose(symmetrical.getPhiD(), coord.getPhiD() + glm::half_pi<double>(), 1e-6));
}