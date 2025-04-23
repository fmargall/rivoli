#include <gtest/gtest.h>

#include "arrays.hpp"
#include "coordinateConversion.hpp"

// Test for the conversion from standard coordinates to Rusinkiewicz coordinates
TEST(CoordinateConversionTest, RoundTripIdentity) {
	// Starting from standard coordinates
    double thetaI   = 0.2;
    double thetaO   = 0.6;
    double deltaPhi = 1.4;

    // Conversion to Rusinkiewicz
    auto rus = standardCoordToRusinkiewiczCoord(thetaI, thetaO, deltaPhi);

    // Inverse conversion
    auto standard = rusinkiewiczCoordToStandardCoord(
        std::get<0>(rus), std::get<1>(rus), std::get<2>(rus));

    // Vérification de l'identité dans la limite numérique
    EXPECT_TRUE(isClose(thetaI  , std::get<0>(standard), 1e-6));
    EXPECT_TRUE(isClose(thetaO  , std::get<1>(standard), 1e-6));
    EXPECT_TRUE(isClose(deltaPhi, std::get<2>(standard), 1e-6));
}