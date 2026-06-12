#pragma once

#include <vectra/vectra.hpp>


namespace rivoli {


template <typename FP, vectra::SIMDLevel level>
struct RusinkiewiczAnglesIsotropic {
    vectra::Vectratype<FP, level> thetaH;
    vectra::Vectratype<FP, level> thetaD;
    vectra::Vectratype<FP, level> phiD;
};

template <typename FP, vectra::SIMDLevel level>
struct RusinkiewiczAnglesAnisotropic {
    vectra::Vectratype<FP, level> thetaH;
    vectra::Vectratype<FP, level> phiH;
    vectra::Vectratype<FP, level> thetaD;
    vectra::Vectratype<FP, level> phiD;
};

template <typename FP, vectra::SIMDLevel level>
struct SphericalAnglesIsotropic {
    vectra::Vectratype<FP, level> thetaI;
    vectra::Vectratype<FP, level> thetaO;
    vectra::Vectratype<FP, level> deltaPhi;
};

template <typename FP, vectra::SIMDLevel level>
struct SphericalAnglesAnisotropic {
    vectra::Vectratype<FP, level> thetaI;
    vectra::Vectratype<FP, level> phiI;
    vectra::Vectratype<FP, level> thetaO;
    vectra::Vectratype<FP, level> phiO;
};


template <typename FP, vectra::SIMDLevel level>
RusinkiewiczAnglesIsotropic<FP, level> sphericalToRusinkiewicz(
    vectra::Vectratype<FP, level> thetaI,
    vectra::Vectratype<FP, level> thetaO,
    vectra::Vectratype<FP, level> deltaPhi)
{
    using vct = vectra::Vectratype<FP, level>;

    const vct sinI  = vct::sin(thetaI)  , cosI  = vct::cos(thetaI);
    const vct sinO  = vct::sin(thetaO)  , cosO  = vct::cos(thetaO);
    const vct sinDP = vct::sin(deltaPhi), cosDP = vct::cos(deltaPhi);

    // Computing h = normalize(omegaI + omegaO)
    // knowing that in our case phiI = 0, with:
    // omegaI = (sinI        ,       0      , cosI)
    // omegaO = (sinO * cosDP,  sinO * sinDP, cosO)
    vct hx = sinI + sinO * cosDP;
    vct hy = sinO * sinDP;
    vct hz = cosI + cosO;

    const vct normH = vct::sqrt(hx * hx + hy * hy + hz * hz);
    hx = hx / normH;
    hy = hy / normH;
    hz = hz / normH;

    const vct cosThetaH = hz;
    vct sinThetaH = vct::sqrt(hx * hx + hy * hy);

    // In this way we avoid bad behaviour such as zero division
    sinThetaH = vct::max(sinThetaH, vct(static_cast<FP>(1e-7)));
    const vct cosPhiH = hx / sinThetaH;
    const vct sinPhiH = hy / sinThetaH;

    // R_z(-phiH) over omegaI = (sinI, 0, cosI)
    const vct tx =  cosPhiH * sinI;
    const vct ty = -sinPhiH * sinI;
    const vct tz =  cosI;

    // R_y(-thetaH) over (tx, ty, tz)
    const vct omegaDx = cosThetaH * tx - sinThetaH * tz;
    const vct omegaDy = ty;
    const vct omegaDz = sinThetaH * tx + cosThetaH * tz;

    // Clamping for better safety
    const vct cosThetaHClamped = vct::min(vct::max(cosThetaH, -vct::one()), vct::one());
    const vct omegaDzClamped   = vct::min(vct::max(omegaDz, -vct::one()), vct::one());

    return {
        vct::acos(cosThetaHClamped), // thetaH
        vct::acos(omegaDzClamped),   // thetaD
        vct::atan2(omegaDy, omegaDx) // phiD ∈ [-π, π]
    };
}

template <typename FP, vectra::SIMDLevel level>
RusinkiewiczAnglesAnisotropic<FP, level> sphericalToRusinkiewicz(
    vectra::Vectratype<FP, level> thetaI,
    vectra::Vectratype<FP, level> phiI,
    vectra::Vectratype<FP, level> thetaO,
    vectra::Vectratype<FP, level> phiO)
{
    using vct = vectra::Vectratype<FP, level>;
}

template <typename FP, vectra::SIMDLevel level>
SphericalAnglesIsotropic<FP, level> rusinkiewiczToSpherical(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{
    using vct = vectra::Vectratype<FP, level>;

    const vct sinThetaH = vct::sin(thetaH), cosThetaH = vct::cos(thetaH);
    const vct sinThetaD = vct::sin(thetaD), cosThetaD = vct::cos(thetaD);
    const vct sinPhiD   = vct::sin(phiD)  , cosPhiD   = vct::cos(phiD);

    // omegaD = (sinThetaD * cosPhiD, sinThetaD * sinPhiD, cosThetaD)
    const vct omegaDx = sinThetaD * cosPhiD;
    const vct omegaDy = sinThetaD * sinPhiD;
    const vct omegaDz = cosThetaD;

    // omegaI = R_z(0) * R_y(thetaH) * omegaD = R_y(thetaH) * omegaD
    // R_y(thetaH) :
    //  x' =  cos(thetaH) * x + sin(thetaH) * z
    //  y' =  y
    //  z' = -sin(thetaH) * x + cos(thetaH) * z
    const vct omegaIx = cosThetaH * omegaDx + sinThetaH * omegaDz;
    const vct omegaIy = omegaDy;
    const vct omegaIz = -sinThetaH * omegaDx + cosThetaH * omegaDz;

    // omegaO = 2 (omegaI · omegaH) omegaH - omegaI
    const vct dotIH = omegaIx * sinThetaH + omegaIz * cosThetaH;
    const vct k     = dotIH + dotIH;
    const vct omegaOx = k * sinThetaH - omegaIx;
    const vct omegaOy = -omegaIy;
    const vct omegaOz = k * cosThetaH - omegaIz;

    // Clamping for better safety
    const vct omegaIzClamped = vct::min(vct::max(omegaIz, -vct::one()), vct::one());
    const vct omegaOzClamped = vct::min(vct::max(omegaOz, -vct::one()), vct::one());

    const vct crossZ = omegaIx * omegaOy - omegaIy * omegaOx;
    const vct dotXY  = omegaIx * omegaOx + omegaIy * omegaOy;

    return {
        vct::acos(omegaIzClamped), // thetaI
        vct::acos(omegaOzClamped), // thetaO
        vct::atan2(crossZ, dotXY)  // deltaPhi ∈ [-π, π]
    };
}

template <typename FP, vectra::SIMDLevel level>
SphericalAnglesAnisotropic<FP, level> rusinkiewiczToSpherical(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> phiH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{
    using vct = vectra::Vectratype<FP, level>;
}

/*
 * 10.1109/TPAMI.2006.170
 */
template <typename FP, vectra::SIMDLevel level>
void rusinkiewiczToZickler(
    vectra::Vectratype<FP, level> thetaH,
    vectra::Vectratype<FP, level> thetaD,
    vectra::Vectratype<FP, level> phiD)
{
    using vct = vectra::Vectratype<FP, level>;

    vct twoPhiD = 2. * phiD;

    vct u = vct::sin(thetaH) * vct::cos(twoPhiD);
    vct v = vct::sin(thetaH) * vct::sin(twoPhiD);
    vct w = 2. * phiD / vct::pi;
}

} // namespace rivoli
