#include <glm/glm.hpp>

#include "loader.hpp"
#include "logger.hpp"

int main() {
	RBFModelvec3 model = RBFModel<glm::vec3>::readFile("../../../brdfManteauAinou.RBFCoeffs", true, 0);

	std::ofstream file("data.txt");
	std::atomic<size_t> completedIterations{ 0 };
	for (size_t thetaID = 0; thetaID <= 1000; thetaID++) {
		for (size_t phiID = 0; phiID <= 500; phiID++) {
			float theta = glm::radians(90.) * thetaID / 1000;
			float phi   = glm::two_pi<float>() * phiID / 500;
			float red   = model.eval(glm::vec2(glm::radians(5.), 0.f), glm::vec2(theta, phi), 0, 0);
			file << theta << " " << phi << " " << red << "\n";

			logger.displayProgressBar(completedIterations.load(std::memory_order_relaxed), 1001*501); // Strangest bug ever: if m_nbClusters is exactly 1041 (as it has already happened once), the progress bar does not appear. It won't happen for 1039, 1040 or 1042.
			completedIterations.fetch_add(1, std::memory_order_relaxed);
		}
	}

	file.close();

	return 0;
}