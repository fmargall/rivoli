#include <gtest/gtest.h>

#include "arrays.hpp"
#include "topology.hpp"

TEST(Topology2DTest, DistanceBetweenPoints) {
    Coordinate2D<double> a(0.2, 0.4);
    Coordinate2D<double> b(0.5, 1.1);
    Topology2D<double> topology;

    auto dist = topology.getDistance(a, b);
    EXPECT_GT(dist, 0.0);
    EXPECT_LT(dist, glm::pi<double>());
}

TEST(Topology2DSymTest, SymmetryRespected) {
    Coordinate2D<double> a(0.2, 0.4);
    Coordinate2D<double> b(0.5, 1.1);
    Topology2DSym<double> topologySym;

    auto direct = Topology2D<double>().getDistance(a, b);
    auto withSym = topologySym.getDistance(a, b);

    EXPECT_LE(withSym, direct);
}

TEST(Topology3DSphRecSymTest, ReciprocalAndSymmetricalMinimization) {
    Coordinate3DSpherical<double> a(0.2, 0.3, 1.4);
    Coordinate3DSpherical<double> b(0.5, 0.6, 2.2);
    Topology3DSphRecSym<double> topology;

    auto dist = topology.getDistance(a, b);
    EXPECT_GT(dist, 0.0);
}