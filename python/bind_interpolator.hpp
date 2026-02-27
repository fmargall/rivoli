#pragma once

#include <nanobind/nanobind.h>

#include <rivoli/rivoli.hpp>


namespace rivoli {


template <typename FP, vectra::SIMDLevel level, size_t dim, bool bilateral, bool reciprocal, KernelType kernelType>
void bindInterpolator(nb::module_& m, const char* name) {

	using KernelClass   = rivoli::KernelSelector<kernelType, FP, level>;
	using TopologyClass = rivoli::TopologySelector<dim, bilateral, reciprocal, FP, level>;

	using InterpolatorClass = Interpolator<KernelClass, TopologyClass>;
	
	nb::class_<InterpolatorClass> cls(m, name);

	if constexpr (dim == 2) {
		cls.def("interpolate", [](const InterpolatorClass& self, FP theta, FP phi) {
			return self.interpolate(theta, phi);
		});
	}
	else if constexpr (dim == 3) {
		cls.def("interpolate", [](const InterpolatorClass& self, FP thetaI, FP thetaO, FP deltaPhi) {
			return self.interpolate(thetaI, thetaO, deltaPhi);
		});
	}
	else if constexpr (dim == 4) {
		cls.def("interpolate", [](const InterpolatorClass& self, FP thetaI, FP phiI, FP thetaO, FP phiO) {
			return self.interpolate(thetaI, phiI, thetaO, phiO);
		});
	}
}


}