#pragma once

#include <Eigen/Dense>

namespace rivoli {


enum class SolverType {
	LLT,
	LDLT,
	QR,
	PartialPivLU,
	FullPivLU
};


}
