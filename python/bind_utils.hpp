#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <rivoli/utils/utils.hpp>


namespace nb = nanobind;


namespace rivoli {


template <typename FP, vectra::SIMDLevel level>
void bindUtils(nb::module_& m) {

    using vct = vectra::Vectratype<FP, level>;

    // For now only the None SIMD level is supported
    if constexpr (level == vectra::SIMDLevel::None) {
        std::string suffix  = "_";
        suffix             += vectra::toString(level);
        suffix             += std::is_same_v<FP, float> ? "_f32" : "_f64";

        std::string nameSphericalToRusinkiewicz = "_spherical_to_rusinkiewicz" + suffix;
        std::string nameRusinkiewiczToSpherical = "_rusinkiewicz_to_spherical" + suffix;

        m.def(nameSphericalToRusinkiewicz.c_str(),
            [](FP thetaI, FP thetaO, FP deltaPhi) {
                  const auto a = sphericalToRusinkiewicz<FP, level>(vct(thetaI), vct(thetaO), vct(deltaPhi));
                  return std::make_tuple(a.thetaH.value, a.thetaD.value, a.phiD.value);
            }, nb::arg("thetaI"), nb::arg("thetaO"), nb::arg("deltaPhi"),
               "Conversion from spherical angles to Rusinkiewicz angles");

        m.def(nameRusinkiewiczToSpherical.c_str(),
            [](FP thetaH, FP thetaD, FP phiD) {
                  const auto a = rusinkiewiczToSpherical<FP, level>(vct(thetaH), vct(thetaD), vct(phiD));
                  return std::make_tuple(a.thetaI.value, a.thetaO.value, a.deltaPhi.value);
            }, nb::arg("thetaH"), nb::arg("thetaD"), nb::arg("phiD"),
               "Convert the Rusinkiewicz angles to spherical angles");
    }
}


} // namespace rivoli
