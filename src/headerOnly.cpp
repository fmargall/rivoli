#include "loader.hpp"

int main() {
	RBFModelvec3 model = RBFModel<glm::vec3>::readFile("brdfManteauAinou.RBFCoeffs", true, 1);

	std::ofstream file("data.txt");
	for (size_t thetaID = 0; thetaID < 100; thetaID++) {
		for (size_t phiID = 0; phiID < 100; phiID++) {
			float theta = glm::radians(90.) * thetaID / 100;
			float phi = glm::two_pi<float>() * phiID / 100;
			glm::vec3 res = model.eval(glm::vec2(glm::radians(7.37), 0.f), glm::vec2(theta, phi));
			float red = res.r;
			file << theta << " " << phi << " " << red << "\n";
		}
	}

	file.close();

	return 0;
}