#pragma once

#include <string>

#include <Eigen/Dense>

#include <tinylogger/tinylogger.hpp>


template <typename FP>
class Regularizer {

public:
    Regularizer(const std::string& solver = "GCV",
                const std::string& method = "identity",
                const FP lambda = static_cast<FP>(0.))
        : _solver(solver), _method(method), _lambda(lambda) {}

    void apply(Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic>& systemMatrix,
         const Eigen::Vector<FP, Eigen::Dynamic>& rightHandVector)
    {
        // Choosing the regularization factor lambda
        if      (_solver == "fixed") {
            // _lambda has already been set by user
        }
        else if (_solver == "GCV")
            _lambda = _optimizeGCV(systemMatrix, rightHandVector);
        else
            LOG_CRITICAL("Unknown regularization solver: ", _solver,
                ". Currently only 'fixed' and 'GCV' are supported.");

        LOG_DEBUG("Regularization factor lambda: ", _lambda,
                  " selected using ", _solver, " solver.");

        // Applying the regularization to the matrix
        if (_method == "identity")
            systemMatrix.diagonal().array() += _lambda;
        else
            LOG_CRITICAL("Unknown regularization method: ", _method,
                         ". Currently only 'identity' is supported.");
    }

private:
    FP _optimizeGCV(
        const Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic>& systemMatrix,
        const Eigen::Vector<FP, Eigen::Dynamic>&                 rightHandVector) const
    {
        const Eigen::Index N = systemMatrix.rows();

        // Eigendecomposition of the (symmetric) system matrix.
        // Computed once: each GCV evaluation then becomes O(N)
        LOG_TRACE("Computing decomposition for GCV optimization..");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic>>
            eigenDecomposition(systemMatrix);

        if (eigenDecomposition.info() != Eigen::Success)
            LOG_CRITICAL("Eigen decomposition failed during GCV optimization.");

        const auto eigenValues = eigenDecomposition.eigenvalues().eval();
        const Eigen::Vector<FP, Eigen::Dynamic> rightHandVectorInEigenBasis =
            eigenDecomposition.eigenvectors().transpose() * rightHandVector;

        // Logarithmic grid bounds, scaled to the system's spectrum
        const FP traceNorm = eigenValues.sum() / static_cast<FP>(N);
        const FP lambdaMin = std::max(static_cast<FP>(1e-14), std::numeric_limits<FP>::epsilon() * traceNorm);
        const FP lambdaMax = traceNorm;

        const size_t gridSize = 200;
        const FP logMin = std::log10(lambdaMin);
        const FP logMax = std::log10(lambdaMax);

        FP bestLambda = lambdaMin;
        FP bestGCV = std::numeric_limits<FP>::infinity();

        for (size_t i = 0; i < gridSize; i++) {
            const FP t      = static_cast<FP>(i) / static_cast<FP>(gridSize - 1);
            const FP lambda = std::pow(static_cast<FP>(10.0), logMin + t * (logMax - logMin));

            // Compute GCV score for the current lambda
            const auto denom    = eigenValues.array() + lambda;
            const FP traceH     = (eigenValues.array() / denom).sum();
            const FP residualSq = ((lambda / denom) * rightHandVectorInEigenBasis.array()).square().sum();

            const FP numerator   = residualSq / static_cast<FP>(N);
            const FP denomTerm   = (static_cast<FP>(N) - traceH) / static_cast<FP>(N);

            // Skip degenerate denominator
            if (denomTerm < std::numeric_limits<FP>::epsilon()) continue;

            const FP gcvScore = numerator / (denomTerm * denomTerm);

            if (gcvScore < bestGCV) {
                bestGCV = gcvScore;
                bestLambda = lambda;
            }
        }

        // Warning if the optimum value sits at the boundary
        if (bestLambda <= lambdaMin * static_cast<FP>(1.01) ||
            bestLambda >= lambdaMax * static_cast<FP>(0.99))
            LOG_WARNING("GCV optimum found near grid boundary (lambda =", bestLambda,
                        "). Consider expanding the search range, for now too narrow.");

        return bestLambda;
    }

    std::string _solver;
    std::string _method;
    FP _lambda;
};
