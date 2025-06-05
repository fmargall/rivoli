#include <glm/glm.hpp>

#include "loader.hpp"
#include "logger.hpp"

int main() {
	std::ofstream file("data.txt");
	std::atomic<size_t> completedIterations{ 0 };

	size_t nbClusters = 1041;
	size_t nbTheta = 10;
	size_t nbPhi = 5;

	size_t nbReplications = 50;

	RBFModelvec3 model = RBFModel<glm::vec3>::readFile("../../../brdfManteauAinou.RBFCoeffs", false, -1);

	for (size_t replication = 0; replication < nbReplications; replication++) {
		
		for (size_t clusterID = 0; clusterID < nbClusters; clusterID++) {

			float theta = glm::radians(5.f);
			float phi = glm::radians(60.f);

			for (size_t channelID = 0; channelID <= 2; channelID++) {
				float red = model.eval(glm::vec2(glm::radians(5.), 0.f), glm::vec2(theta, phi), clusterID, channelID);
				file << theta << " " << phi << " " << red << "\n";
				completedIterations.fetch_add(1, std::memory_order_relaxed);
				logger.displayProgressBar(completedIterations.load(std::memory_order_relaxed), nbClusters * nbReplications * 3);
			}
		}
	}

	file.close();

	return 0;
}