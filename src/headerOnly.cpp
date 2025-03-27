#include "loader.hpp"

int main() {
	RBFModelvec3 model = RBFModel<glm::vec3>::readFile("brdfManteauAinou.RBFCoeffs", true, 30);
	glm::vec3 res = model.eval(glm::vec2(glm::radians(5.)), glm::vec2(glm::radians(5.), 0.0));
	LOG_INFO("res: ", res.x, " ", res.y, " ", res.z);

	return 0;
}