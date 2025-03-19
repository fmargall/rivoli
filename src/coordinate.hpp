#pragma once

#include "coordinateConversion.hpp"

class Coordinate {
public:
	// Adding a virtual constructor makes Coordinate polymorphic,
	// and so allows to use dynamic_cast in the topology distance
	virtual ~Coordinate() = default;
};

template <typename FloatingPrecision>
class Coordinate2D : public Coordinate {
public:
	Coordinate2D(const FloatingPrecision& theta, const FloatingPrecision& phi) : m_theta(theta), m_phi(phi) {}

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
		: m_thetaI(thetaI), m_thetaO(thetaO), m_deltaPhi(deltaPhi) {}

	FloatingPrecision getThetaI()   const { return m_thetaI; }
	FloatingPrecision getThetaO()   const { return m_thetaO; }
	FloatingPrecision getDeltaPhi() const { return m_deltaPhi; }

	Coordinate3DSpherical getBilateralSymmetrical() const {
		return Coordinate3DSpherical(m_thetaI, m_thetaO, glm::two_pi<FloatingPrecision>() - m_deltaPhi);
	}

	Coordinate3DSpherical getReciprocal() const {
		return Coordinate3DSpherical(m_thetaO, m_thetaI, glm::two_pi<FloatingPrecision>() - m_deltaPhi);
	}

private:
	FloatingPrecision m_thetaI, m_thetaO, m_deltaPhi;

};

template <typename FloatingPrecision>
class Coordinate3DRusinkiewicz : public Coordinate {
public:
	Coordinate3DRusinkiewicz(const FloatingPrecision& thetaH,
							 const FloatingPrecision& thetaD,
							 const FloatingPrecision& phiD)
		: m_thetaH(thetaH), m_thetaD(thetaD), m_phiD(phiD) {}

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