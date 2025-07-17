#include <glm/glm.hpp>

#include "loader.hpp"
#include "logger.hpp"

int main() {
	std::ofstream file("data.txt");
	std::atomic<size_t> completedIterations{ 0 };

	size_t clusterID = 400;
	size_t nbTheta = 90;
	size_t nbPhi = 360;

	RBFModelvec3 model = RBFModel<glm::vec3>::readFile("../../../brdfAlbatre2_thetaMax60_4.RBFCoeffs", true, -1);

	float thetaI = model.getMeanThetaI(clusterID);
	float phiI   = glm::radians(60.f);

	LOG_INFO("MeanThetaI: ", thetaI);

	for (size_t thetaID = 0; thetaID < nbTheta; thetaID++) {
		float thetaO = (float)thetaID * glm::half_pi<float>() / (float)nbTheta;

		for (size_t phiID = 0; phiID < nbPhi; phiID++) {
			float phiO = (float)phiID * glm::two_pi<float>() / (float)nbPhi;

			float red = model.eval(glm::vec2(thetaI, phiI), glm::vec2(thetaO, phiO), clusterID, 0);
			file << thetaO << " " << phiO << " " << red << "\n";

			completedIterations.fetch_add(1, std::memory_order_relaxed);
			logger.displayProgressBar(completedIterations.load(std::memory_order_relaxed), nbTheta * nbPhi - 99);
		}
	}

	file.close();

	return 0;
}