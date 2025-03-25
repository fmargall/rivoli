#include "rivoli.hpp"

int main() {
	RBFModelvec3 model = RBFModel<glm::vec3>::readFile("output.RBFCoeffs", true);
	LOG_INFO("Parameterisation: ", model.m_parameterisation);
	LOG_INFO("Number of dimensions: ", model.m_topology->getDimension());
	glm::vec3 result = model.eval(glm::vec2(0.0f), glm::vec2(0.9f * glm::half_pi<float>()), 27);
	LOG_INFO("BRDF value: ", result.x, " ", result.y, " ", result.z);
	return 0;
}