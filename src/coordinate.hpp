#pragma once

#include "coordinateConversion.hpp"

class Coordinate {
public:
	// Adding a virtual constructor makes Coordinate polymorphic,
	// and so allows to use dynamic_cast in the topology distance
	virtual ~Coordinate() = default;

	virtual std::unique_ptr<Coordinate> clone() const {
		LOG_CRITICAL("Cannot clone a base Coordinate object.");
		return nullptr;
	}

};

class GrazingCoordinate : public Coordinate {
public:
	GrazingCoordinate(const size_t& hemisphere) : hemisphere(hemisphere) {}

	std::unique_ptr<Coordinate> clone() const override {
		return std::make_unique<GrazingCoordinate>(*this);
	}

	size_t getHemisphere() const { return hemisphere; }

private:
	const size_t hemisphere;
};

template <typename FloatingPrecision>
class Coordinate2D : public Coordinate {
public:
	Coordinate2D(const FloatingPrecision& theta, const FloatingPrecision& phi) : m_theta(theta), m_phi(phi) {
		//if (theta > glm::half_pi<FloatingPrecision>())
		//	LOG_ERR("Theta value is greater than pi/2: ", theta);
	}

	std::unique_ptr<Coordinate> clone() const override {
		return std::make_unique<Coordinate2D>(*this);
	}

	Coordinate2D getBilateralSymmetrical() const {
		return Coordinate2D(m_theta, glm::two_pi<FloatingPrecision>() - m_phi);
	}

	FloatingPrecision getTheta() const { return m_theta; }
	FloatingPrecision getPhi()   const { return m_phi; }

private:
	FloatingPrecision m_theta, m_phi;
};

template <typename FloatingPrecision>
class Coordinate3DSpherical : public Coordinate {
public:
	Coordinate3DSpherical(const FloatingPrecision& thetaI, 
		                  const FloatingPrecision& thetaO,
		                  const FloatingPrecision& deltaPhi)
		: m_thetaI(thetaI), m_thetaO(thetaO), m_deltaPhi(deltaPhi) {
		//if (thetaI > glm::half_pi<FloatingPrecision>())
		//	LOG_ERR("ThetaI value is greater than pi/2: ", thetaI);
		//if (thetaO > glm::half_pi<FloatingPrecision>())
		//	LOG_ERR("ThetaO value is greater than pi/2: ", thetaO);
	}

	std::unique_ptr<Coordinate> clone() const override {
		return std::make_unique<Coordinate3DSpherical>(*this);
	}

	FloatingPrecision getThetaI()   const { return m_thetaI; }
	FloatingPrecision getThetaO()   const { return m_thetaO; }
	FloatingPrecision getDeltaPhi() const { return m_deltaPhi; }

	Coordinate3DSpherical getBilateralSymmetrical() const {
		return Coordinate3DSpherical(m_thetaI, m_thetaO, glm::two_pi<FloatingPrecision>() - m_deltaPhi);
	}

	Coordinate3DSpherical getReciprocal() const {
		return Coordinate3DSpherical(m_thetaO, m_thetaI, glm::two_pi<FloatingPrecision>() - m_deltaPhi);
	}

	void setThetaI  (const FloatingPrecision& thetaI  ) { m_thetaI   = thetaI; }
	void setThetaO  (const FloatingPrecision& thetaO  ) { m_thetaO   = thetaO; }
	void setDeltaPhi(const FloatingPrecision& deltaPhi) { m_deltaPhi = deltaPhi; }

private:
	FloatingPrecision m_thetaI, m_thetaO, m_deltaPhi;

};

template <typename FloatingPrecision>
class Coordinate3DRusinkiewicz : public Coordinate {
public:
	Coordinate3DRusinkiewicz(const FloatingPrecision& thetaH,
							 const FloatingPrecision& thetaD,
							 const FloatingPrecision& phiD)
		: m_thetaH(thetaH), m_thetaD(thetaD), m_phiD(phiD) {
		if (thetaH > glm::half_pi<FloatingPrecision>())
			LOG_ERR("ThetaI value is greater than pi/2: ", thetaH);
		if (thetaD > glm::half_pi<FloatingPrecision>())
			LOG_ERR("ThetaO value is greater than pi/2: ", thetaD);
	}

	std::unique_ptr<Coordinate> clone() const override {
		return std::make_unique<Coordinate3DRusinkiewicz>(*this);
	}

	explicit operator Coordinate3DSpherical<FloatingPrecision>() const {
		auto [thetaI, thetaO, deltaPhi] = rusinkiewiczCoordToStandardCoord(m_thetaH, m_thetaD, m_phiD);
		return Coordinate3DSpherical<FloatingPrecision>(thetaI, thetaO, deltaPhi);
	}

	FloatingPrecision getThetaH() const { return m_thetaH; }
	FloatingPrecision getThetaD() const { return m_thetaD; }
	FloatingPrecision getPhiD()   const { return m_phiD; }

	Coordinate3DRusinkiewicz getBilateralSymmetrical() const {
		return Coordinate3DRusinkiewicz(m_thetaH, m_thetaD, m_phiD + glm::half_pi<FloatingPrecision>());
	}

	Coordinate3DRusinkiewicz getReciprocal() const {
		return Coordinate3DRusinkiewicz(m_thetaH, m_thetaD, m_phiD + glm::pi<FloatingPrecision>());
	}

private:
	FloatingPrecision m_thetaH, m_thetaD, m_phiD;

};

template <typename FloatingPrecision>
class Coordinate4DSpherical : public Coordinate {
public:

private:

};

template <typename FloatingPrecision>
class Coordinate4DRusinkiewicz : public Coordinate {
public:

private:
};