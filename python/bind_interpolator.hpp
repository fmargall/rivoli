#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <rivoli/rivoli.hpp>


namespace nb = nanobind;


namespace rivoli {


template <typename FP, vectra::SIMDLevel level, size_t dim, bool bilateral, bool reciprocal, KernelType kernelType>
void bindInterpolator(nb::module_& m) {

	using KernelClass   = rivoli::KernelSelector<kernelType, FP, level>;
	using TopologyClass = rivoli::TopologySelector<dim, bilateral, reciprocal, FP, level>;

	using InterpolatorClass = Interpolator<KernelClass, TopologyClass>;

	// Definition of the interpolator name
	std::string name = "_Interpolator_";

	name += std::string(TopologyClass::name);
	name += "_";

	name += std::string(KernelClass::name);
	name += "_";

	name += vectra::toString(level);

	name += std::is_same_v<FP, float> ? "_f32" : "_f64";
	
	// Class instantiation
	nb::class_<InterpolatorClass> cls(m, name.c_str());

	// Class constructor instantiation
	cls.def("__init__", [](InterpolatorClass* self,
		const nb::ndarray<FP, nb::shape<-1, dim>, nb::c_contig>& coordinatesFromPython,
		const nb::ndarray<FP, nb::shape<-1>,      nb::c_contig>& coefficientsFromPython
		) {
			size_t n = coordinatesFromPython.shape(0);

			std::array<std::vector<FP>, dim> coordinates;
			std::vector<FP> coefficients(n);

			for (size_t d = 0; d < dim; ++d)
				coordinates[d].resize(n);

			for (size_t i = 0; i < n; ++i) {
				for (size_t d = 0; d < dim; ++d)
					coordinates[d][i] = coordinatesFromPython(i, d);

				coefficients[i] = coefficientsFromPython(i);
			}

			KernelClass   kernel{};
			TopologyClass topology{};

			new (self) InterpolatorClass(coordinates, coefficients, kernel, topology);

		});

	// interpolate function instantiation
	if      constexpr (dim == 2) {
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